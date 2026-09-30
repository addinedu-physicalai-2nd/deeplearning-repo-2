"""Robot UDP 영상 수신 및 JPEG 프레임 재조립 모듈."""

import queue
import socket
import struct
import time

from web_server.network_modules.connection_events import (
    ConnectionEventLogger,
)


def _put_latest(frame_buffer, frame):
    if frame_buffer.full():
        try:
            frame_buffer.get_nowait()
            frame_buffer.task_done()
        except queue.Empty:
            pass
    frame_buffer.put_nowait(frame)


def _remove_expired_frames(pending_frames, now, frame_timeout):
    expired = []
    for key, frame in list(pending_frames.items()):
        if now - frame["updated_at"] > frame_timeout:
            expired.append(
                (
                    key,
                    len(frame["chunks"]),
                    frame["total_chunks"],
                )
            )
            del pending_frames[key]
    return expired


def _receive_udp_frame(
    sock,
    pending_frames,
    header_format,
    frame_timeout,
    on_valid_packet,
    on_receive_failure,
):
    header_size = struct.calcsize(header_format)

    while True:
        packet, robot_address = sock.recvfrom(65535)
        now = time.monotonic()
        expired_frames = _remove_expired_frames(
            pending_frames,
            now,
            frame_timeout,
        )
        for (address, frame_id), received, total in expired_frames:
            on_receive_failure(
                f"incomplete frame {frame_id} from {address[0]}:{address[1]} "
                f"({received}/{total} chunks)"
            )

        if len(packet) <= header_size:
            on_receive_failure(
                f"short UDP packet from {robot_address[0]}:"
                f"{robot_address[1]} ({len(packet)} bytes)"
            )
            continue

        frame_id, chunk_idx, total_chunks = struct.unpack(
            header_format,
            packet[:header_size],
        )
        if total_chunks == 0 or chunk_idx >= total_chunks:
            on_receive_failure(
                f"invalid chunk header from {robot_address[0]}:"
                f"{robot_address[1]} (chunk={chunk_idx}, total={total_chunks})"
            )
            continue

        on_valid_packet(robot_address, now)

        key = (robot_address, frame_id)
        frame = pending_frames.get(key)
        if frame is None or frame["total_chunks"] != total_chunks:
            frame = {
                "total_chunks": total_chunks,
                "chunks": {},
                "updated_at": now,
            }
            pending_frames[key] = frame

        frame["chunks"][chunk_idx] = packet[header_size:]
        frame["updated_at"] = now

        if len(frame["chunks"]) != total_chunks:
            continue

        image_data = b"".join(
            frame["chunks"][index] for index in range(total_chunks)
        )
        del pending_frames[key]

        if not image_data.startswith(b"\xff\xd8"):
            on_receive_failure(
                f"invalid JPEG frame {frame_id} from "
                f"{robot_address[0]}:{robot_address[1]}"
            )
            continue

        return frame_id, image_data, robot_address


def receive_frames(
    frame_buffer,
    state,
    logger,
    stop_event,
    udp_host,
    udp_port,
    header_format="!HBB",
    frame_timeout=1.0,
    disconnect_timeout=3.0,
):
    """UDP 프레임을 수신해 AI 처리 대기 버퍼에 저장한다."""
    pending_frames = {}
    last_packet_at = None
    connection = ConnectionEventLogger(
        "Robot",
        state,
        "robot_connected",
        logger,
    )

    def on_valid_packet(robot_address, received_at):
        nonlocal last_packet_at
        last_packet_at = received_at
        connection.connection_succeeded(
            f"{robot_address[0]}:{robot_address[1]}"
        )
        state.update(robot_error=None)

    def on_receive_failure(error):
        connection.receive_failed(error)
        state.update(robot_error=str(error))

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((udp_host, udp_port))
            sock.settimeout(0.5)

            while not stop_event.is_set():
                try:
                    frame_id, image_data, robot_address = _receive_udp_frame(
                        sock,
                        pending_frames,
                        header_format,
                        frame_timeout,
                        on_valid_packet,
                        on_receive_failure,
                    )
                except socket.timeout:
                    now = time.monotonic()
                    if (
                        connection.connected
                        and last_packet_at is not None
                        and now - last_packet_at >= disconnect_timeout
                    ):
                        reason = (
                            f"no UDP packet for {disconnect_timeout:.1f}s"
                        )
                        connection.connection_lost(reason)
                        state.update(robot_error=reason)
                        last_packet_at = None
                    continue

                _put_latest(frame_buffer, (frame_id, image_data))
                state.update(
                    received_frame_id=frame_id,
                    robot_address=robot_address[0],
                    buffer_size=frame_buffer.qsize(),
                )
    except OSError as error:
        if not stop_event.is_set():
            if connection.connected:
                connection.connection_lost(error)
            else:
                connection.receive_failed(error)
            state.update(robot_error=str(error))
