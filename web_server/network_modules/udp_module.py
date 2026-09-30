"""Robot UDP 영상 수신 및 JPEG 프레임 재조립 모듈."""

import queue
import socket
import struct
import time


def _put_latest(frame_buffer, frame):
    if frame_buffer.full():
        try:
            frame_buffer.get_nowait()
            frame_buffer.task_done()
        except queue.Empty:
            pass
    frame_buffer.put_nowait(frame)


def _remove_expired_frames(pending_frames, now, frame_timeout):
    for key, frame in list(pending_frames.items()):
        if now - frame["updated_at"] > frame_timeout:
            del pending_frames[key]


def _receive_udp_frame(
    sock,
    pending_frames,
    header_format,
    frame_timeout,
    log_first_packet,
    logger,
):
    header_size = struct.calcsize(header_format)

    while True:
        packet, robot_address = sock.recvfrom(65535)
        now = time.monotonic()
        _remove_expired_frames(pending_frames, now, frame_timeout)

        if len(packet) <= header_size:
            continue

        frame_id, chunk_idx, total_chunks = struct.unpack(
            header_format,
            packet[:header_size],
        )
        if total_chunks == 0 or chunk_idx >= total_chunks:
            continue

        log_first_packet(robot_address)

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
            logger.warning("Ignored invalid JPEG frame from %s", robot_address)
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
):
    """UDP 프레임을 수신해 AI 처리 대기 버퍼에 저장한다."""
    pending_frames = {}
    first_packet_logged = False

    def log_first_packet(robot_address):
        nonlocal first_packet_logged
        if first_packet_logged:
            return
        first_packet_logged = True
        logger.info(
            "UDP communication active: first packet received from %s:%s",
            robot_address[0],
            robot_address[1],
        )

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
                    log_first_packet,
                    logger,
                )
            except socket.timeout:
                continue
            except OSError:
                if not stop_event.is_set():
                    logger.exception("UDP receiver stopped unexpectedly")
                return

            _put_latest(frame_buffer, (frame_id, image_data))
            state.update(
                received_frame_id=frame_id,
                robot_address=robot_address[0],
                buffer_size=frame_buffer.qsize(),
            )
