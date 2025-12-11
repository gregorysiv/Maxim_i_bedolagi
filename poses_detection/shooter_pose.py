import numpy as np
from poses_detection.calculate_angle import calculate_angle


def check_shooter_pose(
    points,
    confs,
    confidence_threshold=0.5,
    elbow_angle_min=60,
    elbow_angle_max=120,
    shoulder_angle_min=150,
    forearm_alignment_threshold=20,
    stability_threshold=15,
):
    """
    Определяет, находится ли человек в позе стрелка (стойка для стрельбы).

    Ключевые признаки позы стрелка:
    1. Согнутая рука с оружием (угол в локте ~90°)
    2. Вытянутая поддерживающая рука
    3. Устойчивая стойка с расставленными ногами
    4. Тело слегка развернуто

    Args:
        points (np.array): Координаты 17 ключевых точек COCO.
        confs (np.array): Уверенность для каждой точки.
        confidence_threshold (float): Порог уверенности для учета точки.
        elbow_angle_min (float): Минимальный угол в локте для согнутой руки.
        elbow_angle_max (float): Максимальный угол в локте для согнутой руки.
        shoulder_angle_min (float): Минимальный угол в плече для вытянутой руки.
        forearm_alignment_threshold (float): Допуск выравнивания предплечья.
        stability_threshold (float): Порог устойчивости стойки.

    Returns:
        str: " (Shooting Stance)" если поза соответствует, иначе "".
    """

    # Индексы ключевых точек COCO
    NOSE = 0
    L_SHOULDER, R_SHOULDER = 5, 6
    L_ELBOW, R_ELBOW = 7, 8
    L_WRIST, R_WRIST = 9, 10
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16

    # 1. Проверка доступности необходимых точек
    required_points_available = (
        confs[L_SHOULDER] > confidence_threshold
        and confs[R_SHOULDER] > confidence_threshold
        and confs[L_ELBOW] > confidence_threshold
        and confs[R_ELBOW] > confidence_threshold
        and confs[L_WRIST] > confidence_threshold
        and confs[R_WRIST] > confidence_threshold
        and confs[L_HIP] > confidence_threshold
        and confs[R_HIP] > confidence_threshold
    )

    if not required_points_available:
        return ""

    # 2. Анализ позы стрелка по нескольким критериям
    criteria_met = 0
    max_criteria = 6

    # КРИТЕРИЙ 1: Одна рука согнута (угол в локте 60-120°)
    # Проверяем обе руки
    left_elbow_angle = calculate_angle(points[L_SHOULDER], points[L_ELBOW], points[L_WRIST])
    right_elbow_angle = calculate_angle(points[R_SHOULDER], points[R_ELBOW], points[R_WRIST])

    bent_arm_found = False
    shooting_arm = None  # Рука, которая согнута (держит оружие)
    support_arm = None  # Рука, которая вытянута (поддерживает)

    # Проверяем левую руку
    if elbow_angle_min < left_elbow_angle < elbow_angle_max:
        bent_arm_found = True
        shooting_arm = "left"
        support_arm = "right"

    # Проверяем правую руку
    if elbow_angle_min < right_elbow_angle < elbow_angle_max:
        bent_arm_found = True
        shooting_arm = "right"
        support_arm = "left"

    if bent_arm_found:
        criteria_met += 1
        print(
            f"  Критерий 1: Согнутая рука найдена ({shooting_arm} рука, угол: {left_elbow_angle if shooting_arm == 'left' else right_elbow_angle:.1f}°)"
        )

    # КРИТЕРИЙ 2: Противоположная рука вытянута (угол в плече > 150°)
    if shooting_arm:
        if support_arm == "left":
            support_shoulder_angle = calculate_angle(
                points[L_ELBOW], points[L_SHOULDER], points[L_HIP]
            )
        else:  # support_arm == "right"
            support_shoulder_angle = calculate_angle(
                points[R_ELBOW], points[R_SHOULDER], points[R_HIP]
            )

        if support_shoulder_angle > shoulder_angle_min:
            criteria_met += 1
            print(
                f"  Критерий 2: Поддерживающая рука вытянута (угол: {support_shoulder_angle:.1f}°)"
            )

    # КРИТЕРИЙ 3: Выравнивание предплечья (горизонтальность)
    if shooting_arm:
        if shooting_arm == "left":
            wrist = points[L_WRIST]
            elbow = points[L_ELBOW]
        else:
            wrist = points[R_WRIST]
            elbow = points[R_ELBOW]

        # Проверяем горизонтальность предплечья (разница по Y минимальна)
        forearm_tilt = abs(wrist[1] - elbow[1])

        if forearm_tilt < forearm_alignment_threshold:
            criteria_met += 1
            print(f"  Критерий 3: Предплечье горизонтально (наклон: {forearm_tilt:.1f} px)")

    # КРИТЕРИЙ 4: Устойчивая стойка (ноги расставлены)
    if confs[L_ANKLE] > confidence_threshold and confs[R_ANKLE] > confidence_threshold:
        stance_width = abs(points[L_ANKLE][0] - points[R_ANKLE][0])
        stance_height = abs(points[L_ANKLE][1] - points[R_ANKLE][1])

        # Широкая и устойчивая стойка
        if stance_width > 50 and stance_height < stability_threshold:
            criteria_met += 1
            print(
                f"  Критерий 4: Устойчивая стойка (ширина: {stance_width:.1f} px, перепад высоты: {stance_height:.1f} px)"
            )

    # КРИТЕРИЙ 5: Разворот тела (плечи не параллельны бедрам)
    if confs[L_HIP] > confidence_threshold and confs[R_HIP] > confidence_threshold:
        # Угол между линией плеч и линией бедер
        shoulders_vector = np.array(points[R_SHOULDER]) - np.array(points[L_SHOULDER])
        hips_vector = np.array(points[R_HIP]) - np.array(points[L_HIP])

        # Вычисляем угол между векторами
        dot_product = np.dot(shoulders_vector, hips_vector)
        shoulders_norm = np.linalg.norm(shoulders_vector)
        hips_norm = np.linalg.norm(hips_vector)

        if shoulders_norm > 0 and hips_norm > 0:
            cos_angle = dot_product / (shoulders_norm * hips_norm)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_between = np.degrees(np.arccos(cos_angle))

            # В стойке стрелка плечи часто развернуты относительно бедер
            if angle_between > 20:  # Значительный разворот
                criteria_met += 1
                print(
                    f"  Критерий 5: Тело развернуто (угол между плечами и бедрами: {angle_between:.1f}°)"
                )

    # КРИТЕРИЙ 6: Положение головы (взгляд вдоль руки)
    if confs[NOSE] > confidence_threshold and shooting_arm:
        if shooting_arm == "left":
            # Проверяем, смотрит ли голова в направлении левой руки
            nose_to_wrist_vector = np.array(points[L_WRIST]) - np.array(points[NOSE])
            nose_to_shoulder_vector = np.array(points[L_SHOULDER]) - np.array(points[NOSE])
        else:
            nose_to_wrist_vector = np.array(points[R_WRIST]) - np.array(points[NOSE])
            nose_to_shoulder_vector = np.array(points[R_SHOULDER]) - np.array(points[NOSE])

        # Вычисляем угол между направлением взгляда и направлением к запястью
        if np.linalg.norm(nose_to_wrist_vector) > 0 and np.linalg.norm(nose_to_shoulder_vector) > 0:
            cos_angle = np.dot(nose_to_wrist_vector, nose_to_shoulder_vector) / (
                np.linalg.norm(nose_to_wrist_vector) * np.linalg.norm(nose_to_shoulder_vector)
            )
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            head_alignment_angle = np.degrees(np.arccos(cos_angle))

            if head_alignment_angle < 45:  # Голова направлена в сторону руки
                criteria_met += 1
                print(
                    f"  Критерий 6: Голова направлена вдоль руки (угол: {head_alignment_angle:.1f}°)"
                )

    # 4. Принятие решения
    # Для позы стрелка требуется выполнение как минимум 4 из 6 критериев
    if criteria_met >= 4:
        stance_type = (
            "Right-handed" if shooting_arm == "right" else "Left-handed" if shooting_arm else "Unknown"
        )
        print(
            f"  Обнаружена поза стрелка: {stance_type} стойка, критериев выполнено: {criteria_met}/{max_criteria}"
        )
        return " (Shooting Stance)"

    # print(f"  Поза стрелка не обнаружена, критериев выполнено: {criteria_met}/{max_criteria}")
    return ""


def check_shooter_pose_advanced(
    points,
    confs,
    confidence_threshold=0.5,
    history=None,
    min_frames_for_confirmation=3,
):
    """
    Расширенная версия детектора позы стрелка с учетом истории кадров.

    Args:
        points (np.array): Координаты ключевых точек.
        confs (np.array): Уверенность для каждой точки.
        confidence_threshold (float): Порог уверенности.
        history (dict | list | None): История предыдущих определений. Списки
            предыдущей реализации автоматически преобразуются в словарь.
        min_frames_for_confirmation (int): Минимальное количество кадров для
            подтверждения изменения состояния.

    Returns:
        tuple: (результат, обновленная история)
    """

    # Поддержка старого формата истории (список булевых значений)
    if history is None or not isinstance(history, dict):
        history = {
            "detections": list(history) if isinstance(history, list) else [],
            "state": False,
            "frames_in_state": 0,
            "confirmed": False,
        }

    # Определяем позу на текущем кадре
    current_result = check_shooter_pose(points, confs, confidence_threshold)
    is_shooter = current_result != ""

    # Добавляем результат в историю и ограничиваем длину окна
    history["detections"].append(is_shooter)
    max_history = max(min_frames_for_confirmation * 3, 1)
    if len(history["detections"]) > max_history:
        history["detections"] = history["detections"][-max_history:]

    # Отслеживаем продолжительность текущего состояния
    if is_shooter == history.get("state", False):
        history["frames_in_state"] += 1
    else:
        history["state"] = is_shooter
        history["frames_in_state"] = 1

    # Рассчитываем стабильность в окне истории
    recent_detections = history["detections"][-min_frames_for_confirmation:]
    detection_rate = sum(recent_detections) / len(recent_detections)

    positive_stable = (
        history["state"]
        and history["frames_in_state"] >= min_frames_for_confirmation
        and detection_rate >= 0.6
    )
    negative_stable = (
        not history["state"]
        and history["frames_in_state"] >= min_frames_for_confirmation
        and detection_rate <= 0.4
    )

    if positive_stable:
        history["confirmed"] = True
    elif negative_stable:
        history["confirmed"] = False

    return (" (Shooting Stance)" if history["confirmed"] else ""), history


def analyze_shooting_stance_details(points, confs, confidence_threshold=0.5):
    """
    Детальный анализ стойки стрелка с дополнительной информацией.

    Returns:
        dict: Детализированная информация о позе стрелка.
    """

    L_SHOULDER, R_SHOULDER = 5, 6
    L_ELBOW, R_ELBOW = 7, 8
    L_WRIST, R_WRIST = 9, 10
    L_HIP, R_HIP = 11, 12
    L_ANKLE, R_ANKLE = 15, 16

    details = {
        "is_shooter": False,
        "shooting_arm": None,
        "stance_type": None,
        "angles": {},
        "stance_metrics": {},
        "confidence": 0.0,
    }

    # Проверяем базовую позу
    result = check_shooter_pose(points, confs, confidence_threshold)
    details["is_shooter"] = result != ""

    if details["is_shooter"]:
        # Определяем, какая рука согнута
        left_elbow_angle = calculate_angle(points[L_SHOULDER], points[L_ELBOW], points[L_WRIST])
        right_elbow_angle = calculate_angle(points[R_SHOULDER], points[R_ELBOW], points[R_WRIST])

        if 60 < left_elbow_angle < 120:
            details["shooting_arm"] = "left"
            details["stance_type"] = "Left-handed"
        elif 60 < right_elbow_angle < 120:
            details["shooting_arm"] = "right"
            details["stance_type"] = "Right-handed"

        # Сохраняем углы
        details["angles"]["left_elbow"] = left_elbow_angle
        details["angles"]["right_elbow"] = right_elbow_angle

        # Измеряем ширину стойки
        if confs[L_ANKLE] > confidence_threshold and confs[R_ANKLE] > confidence_threshold:
            stance_width = abs(points[L_ANKLE][0] - points[R_ANKLE][0])
            details["stance_metrics"]["width"] = stance_width
            details["stance_metrics"]["stance_score"] = min(stance_width / 100, 1.0)

        # Вычисляем общую уверенность
        angles_in_range = 0
        total_angles = 0

        if 60 < details["angles"]["left_elbow"] < 120:
            angles_in_range += 1
        total_angles += 1

        if 60 < details["angles"]["right_elbow"] < 120:
            angles_in_range += 1
        total_angles += 1

        details["confidence"] = angles_in_range / total_angles

    return details
