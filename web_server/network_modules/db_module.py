"""AI 서버의 JSON 분석 결과를 기존 MySQL 테이블에 저장한다."""

import queue
from decimal import Decimal

import mysql.connector


FRAME_SUMMARY_INSERT = """
    INSERT INTO `frame_summary` (
        `frame`, `pothole_count`, `people_count`,
        `trafficCone_count`, `car_count`, `lane_count`
    ) VALUES (%s, %s, %s, %s, %s, %s)
"""

OBJECT_INSERTS = {
    "pothole": """
        INSERT INTO `pothole` (
            `frame_summary_id`, `frame`, `sequence_id`,
            `confidence`, `b_box`, `risk_level`
        ) VALUES (%s, %s, %s, %s, %s, %s)
    """,
    "people": """
        INSERT INTO `people` (
            `frame_summary_id`, `frame`, `sequence_id`,
            `confidence`, `b_box`
        ) VALUES (%s, %s, %s, %s, %s)
    """,
    "trafficCone": """
        INSERT INTO `trafficCone` (
            `frame_summary_id`, `frame`, `sequence_id`,
            `confidence`, `b_box`
        ) VALUES (%s, %s, %s, %s, %s)
    """,
    "car": """
        INSERT INTO `car` (
            `frame_summary_id`, `frame`, `sequence_id`,
            `confidence`, `b_box`
        ) VALUES (%s, %s, %s, %s, %s)
    """,
}

LANE_INSERT = """
    INSERT INTO `lane` (
        `frame_summary_id`, `frame`, `class_id`, `confidence`
    ) VALUES (%s, %s, %s, %s)
"""

DETECTION_NAMES = ("pothole", "people", "trafficCone", "car", "lane")

DASHBOARD_TABLES = {
    "frame_summary": (
        "id",
        "frame",
        "pothole_count",
        "people_count",
        "trafficCone_count",
        "car_count",
        "lane_count",
        "created_at",
    ),
    "pothole": (
        "id",
        "frame_summary_id",
        "frame",
        "sequence_id",
        "confidence",
        "b_box",
        "risk_level",
    ),
    "people": (
        "id",
        "frame_summary_id",
        "frame",
        "sequence_id",
        "confidence",
        "b_box",
    ),
    "trafficCone": (
        "id",
        "frame_summary_id",
        "frame",
        "sequence_id",
        "confidence",
        "b_box",
    ),
    "car": (
        "id",
        "frame_summary_id",
        "frame",
        "sequence_id",
        "confidence",
        "b_box",
    ),
    "lane": (
        "id",
        "frame_summary_id",
        "frame",
        "class_id",
        "confidence",
    ),
}

DELETE_ORDER = (
    "pothole",
    "people",
    "trafficCone",
    "car",
    "lane",
    "frame_summary",
)


def format_b_box(b_box):
    """좌표 객체를 DB 규격인 [(x1,y1),(x2,y2)] 문자열로 바꾼다."""
    if not isinstance(b_box, dict):
        raise ValueError("b_box must be a JSON object")

    required = ("x_min", "y_min", "x_max", "y_max")
    if any(key not in b_box for key in required):
        raise ValueError("b_box is missing coordinate values")

    return (
        f"[({b_box['x_min']},{b_box['y_min']}),"
        f"({b_box['x_max']},{b_box['y_max']})]"
    )


def _extract_detections(ai_result):
    if not isinstance(ai_result, dict):
        raise ValueError("AI result must be a JSON object")

    detections = ai_result.get("detections")
    if not isinstance(detections, dict):
        raise ValueError("AI result must contain a detections object")

    extracted = {}
    for name in DETECTION_NAMES:
        items = detections.get(name, [])
        if not isinstance(items, list):
            raise ValueError(f"detections.{name} must be a list")
        extracted[name] = items
    return extracted


def _insert_frame_summary(cursor, frame_id, detections):
    cursor.execute(
        FRAME_SUMMARY_INSERT,
        (
            frame_id,
            len(detections["pothole"]),
            len(detections["people"]),
            len(detections["trafficCone"]),
            len(detections["car"]),
            len(detections["lane"]),
        ),
    )
    return cursor.lastrowid


def _object_rows(frame_summary_id, frame_id, items, include_risk=False):
    rows = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each detection must be a JSON object")

        row = (
            frame_summary_id,
            frame_id,
            item["sequence_id"],
            item["confidence"],
            format_b_box(item["b_box"]),
        )
        if include_risk:
            risk_level = item.get("risk_level")
            if risk_level is None:
                risk_level = item.get("level")
            row += (risk_level,)
        rows.append(row)
    return rows


def _insert_object_detections(
    cursor,
    frame_summary_id,
    frame_id,
    detections,
):
    for name in ("pothole", "people", "trafficCone", "car"):
        rows = _object_rows(
            frame_summary_id,
            frame_id,
            detections[name],
            include_risk=name == "pothole",
        )
        if rows:
            cursor.executemany(OBJECT_INSERTS[name], rows)


def _insert_lane_detections(
    cursor,
    frame_summary_id,
    frame_id,
    lanes,
):
    rows = []
    for lane in lanes:
        if not isinstance(lane, dict):
            raise ValueError("each lane detection must be a JSON object")
        rows.append(
            (
                frame_summary_id,
                frame_id,
                lane["class_id"],
                lane["confidence"],
            )
        )

    if rows:
        cursor.executemany(LANE_INSERT, rows)


def insert_ai_result(connection, frame_id, ai_result):
    """한 프레임의 요약과 상세 결과를 하나의 트랜잭션으로 저장한다."""
    detections = _extract_detections(ai_result)
    cursor = connection.cursor()

    try:
        frame_summary_id = _insert_frame_summary(
            cursor,
            frame_id,
            detections,
        )
        _insert_object_detections(
            cursor,
            frame_summary_id,
            frame_id,
            detections,
        )
        _insert_lane_detections(
            cursor,
            frame_summary_id,
            frame_id,
            detections["lane"],
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()


def _serialize_db_value(value):
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat(sep=" ", timespec="seconds")
        except TypeError:
            return value.isoformat()
    return value


def fetch_dashboard_summary(db_config):
    """frame_summary를 기준으로 누적 검출 개수를 조회한다."""
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT
                COUNT(*) AS frame_count,
                COALESCE(SUM(`pothole_count`), 0) AS pothole_count,
                COALESCE(SUM(`people_count`), 0) AS people_count,
                COALESCE(SUM(`trafficCone_count`), 0) AS trafficCone_count,
                COALESCE(SUM(`car_count`), 0) AS car_count,
                COALESCE(SUM(`lane_count`), 0) AS lane_count
            FROM `frame_summary`
            """
        )
        row = cursor.fetchone()
        return {
            key: _serialize_db_value(value)
            for key, value in row.items()
        }
    finally:
        cursor.close()
        connection.close()


def delete_all_detection_data(
    db_config,
    connection_factory=mysql.connector.connect,
):
    """Dashboard에서 사용하는 모든 테이블의 데이터를 삭제한다."""
    connection = connection_factory(**db_config)
    cursor = connection.cursor()
    deleted = {}
    try:
        for table_name in DELETE_ORDER:
            cursor.execute(f"DELETE FROM `{table_name}`")
            deleted[table_name] = max(cursor.rowcount, 0)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()

    return {
        "deleted": deleted,
        "deleted_rows": sum(deleted.values()),
    }


def fetch_table_page(
    db_config,
    table_name,
    page=1,
    page_size=20,
    frame_id=None,
):
    """허용된 테이블을 최신 행부터 페이지 단위로 조회한다."""
    if table_name not in DASHBOARD_TABLES:
        raise ValueError(f"unsupported table: {table_name}")

    page = max(1, int(page))
    page_size = min(100, max(1, int(page_size)))
    columns = DASHBOARD_TABLES[table_name]
    column_sql = ", ".join(f"`{column}`" for column in columns)
    filter_sql = ""
    filter_params = ()
    if frame_id is not None:
        filter_sql = " WHERE `frame` = %s"
        filter_params = (int(frame_id),)

    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            f"SELECT COUNT(*) AS `total` FROM `{table_name}`{filter_sql}",
            filter_params,
        )
        total = cursor.fetchone()["total"]
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = min(page, total_pages)
        offset = (page - 1) * page_size
        cursor.execute(
            f"SELECT {column_sql} FROM `{table_name}`{filter_sql} "
            "ORDER BY `id` DESC LIMIT %s OFFSET %s",
            filter_params + (page_size, offset),
        )
        rows = [
            {
                key: _serialize_db_value(value)
                for key, value in row.items()
            }
            for row in cursor.fetchall()
        ]
    finally:
        cursor.close()
        connection.close()

    return {
        "table": table_name,
        "columns": columns,
        "rows": rows,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
    }


def store_ai_results(
    db_buffer,
    state,
    logger,
    stop_event,
    db_config,
    retry_interval=5.0,
    operation_lock=None,
):
    """DB에 연결하고 큐로 전달된 AI 결과를 순서대로 저장한다."""
    connection = None
    while not stop_event.is_set():
        if connection is None:
            try:
                connection = mysql.connector.connect(**db_config)
                state.update(db_connected=True, db_error=None)
                logger.info(
                    "MySQL connected: %s:%s/%s",
                    db_config["host"],
                    db_config["port"],
                    db_config["database"],
                )
            except Exception as error:
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass
                connection = None
                state.update(db_connected=False, db_error=str(error))
                logger.error("Failed to connect to MySQL: %s", error)
                stop_event.wait(retry_interval)
                continue

        if operation_lock is not None:
            operation_lock.acquire()
        try:
            try:
                frame_id, ai_result = db_buffer.get(timeout=0.5)
            except queue.Empty:
                continue

            state.update(db_buffer_size=db_buffer.qsize())
            try:
                insert_ai_result(connection, frame_id, ai_result)
                state.update(
                    db_connected=True,
                    db_frame_id=frame_id,
                    db_error=None,
                )
                logger.info("Frame %s stored in MySQL", frame_id)
            except (KeyError, TypeError, ValueError) as error:
                state.update(db_error=f"frame {frame_id}: {error}")
                logger.error("Invalid DB data for frame %s: %s", frame_id, error)
            except Exception as error:
                state.update(
                    db_connected=False,
                    db_error=f"frame {frame_id}: {error}",
                )
                logger.error("Failed to store frame %s in MySQL: %s", frame_id, error)
                try:
                    connection.close()
                except Exception:
                    pass
                connection = None
            finally:
                db_buffer.task_done()
                state.update(db_buffer_size=db_buffer.qsize())
        finally:
            if operation_lock is not None:
                operation_lock.release()

    if connection is not None:
        connection.close()
