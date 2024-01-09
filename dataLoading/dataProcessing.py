def load_data_from_file(filename):
    """
    Load sEMG data and labels from the given file.
    """
    data = []
    labels = []

    with open(filename, 'r') as file:
        for line in file:
            # Each line is in the format "GraspType,sEMGValue"
            grasp_type, value = line.strip().split(',')
            data.append(float(value))
            labels.append(grasp_type)

    return data, labels

def segment_data(data, labels, window_size):
    """
    Segment the continuous data into windows.
    """
    segmented_data = []
    segmented_labels = []

    for i in range(0, len(data) - window_size + 1, window_size):
        segmented_data.append(data[i:i+window_size])
        # For simplicity, we'll label the segment with the most common label in the window
        segmented_labels.append(max(set(labels[i:i+window_size]), key=labels[i:i+window_size].count))

    return segmented_data, segmented_labels
