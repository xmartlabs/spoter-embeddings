
_BODY_KEYPOINT_MAPPING = {
    "nose": "nose",
    "left_eye": "leftEye",
    "right_eye": "rightEye",
    "left_ear": "leftEar",
    "right_ear": "rightEar",
    "left_shoulder": "leftShoulder",
    "right_shoulder": "rightShoulder",
    "left_elbow": "leftElbow",
    "right_elbow": "rightElbow",
    "left_wrist": "leftWrist",
    "right_wrist": "rightWrist"
}

_HAND_KEYPOINT_MAPPING = {
    "wrist": "wrist",
    "index_finger_tip": "indexTip",
    "index_finger_dip": "indexDIP",
    "index_finger_pip": "indexPIP",
    "index_finger_mcp": "indexMCP",
    "middle_finger_tip": "middleTip",
    "middle_finger_dip": "middleDIP",
    "middle_finger_pip": "middlePIP",
    "middle_finger_mcp": "middleMCP",
    "ring_finger_tip": "ringTip",
    "ring_finger_dip": "ringDIP",
    "ring_finger_pip": "ringPIP",
    "ring_finger_mcp": "ringMCP",
    "pinky_tip": "littleTip",
    "pinky_dip": "littleDIP",
    "pinky_pip": "littlePIP",
    "pinky_mcp": "littleMCP",
    "thumb_tip": "thumbTip",
    "thumb_ip": "thumbIP",
    "thumb_mcp": "thumbMP",
    "thumb_cmc": "thumbCMC"
}


def map_blazepose_keypoint(column):
    #  Remove _x, _y suffixes
    suffix = column[-2:].upper()
    column = column[:-2]

    if column.startswith("left_hand_"):
        hand = "left"
        finger_name = column[10:]
    elif column.startswith("right_hand_"):
        hand = "right"
        finger_name = column[11:]
    else:
        if column not in _BODY_KEYPOINT_MAPPING:
            return None
        mapped = _BODY_KEYPOINT_MAPPING[column]
        return mapped + suffix

    if finger_name not in _HAND_KEYPOINT_MAPPING:
        return None
    mapped = _HAND_KEYPOINT_MAPPING[finger_name]
    return f"{mapped}_{hand}{suffix}"


def map_blazepose_df(df):
    to_drop = []
    renamings = {}
    for column in df.columns:
        mapped_column = map_blazepose_keypoint(column)
        if mapped_column:
            renamings[column] = mapped_column
        else:
            to_drop.append(column)
    df = df.rename(columns=renamings)

    for index, row in df.iterrows():

        sequence_size = len(row["leftEar_Y"])
        lsx = row["leftShoulder_X"]
        rsx = row["rightShoulder_X"]
        lsy = row["leftShoulder_Y"]
        rsy = row["rightShoulder_Y"]
        neck_x = []
        neck_y = []
        # Treat each element of the sequence (analyzed frame) individually
        for sequence_index in range(sequence_size):
            neck_x.append((float(lsx[sequence_index]) + float(rsx[sequence_index])) / 2)
            neck_y.append((float(lsy[sequence_index]) + float(rsy[sequence_index])) / 2)
        df.loc[index, "neck_X"] = str(neck_x)
        df.loc[index, "neck_Y"] = str(neck_y)

    df.drop(columns=to_drop, inplace=True)
    return df
