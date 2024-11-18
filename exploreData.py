import numpy as np
import os
import glob
import random
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
from collections import Counter, defaultdict

def analyze_class_distribution(image_dir):
    """
    Function to analyze class variability and imbalance in segmentation masks.
    It checks the presence of each class label in each scan and calculates
    the proportion of each label across the entire dataset.
    """
    class_presence = defaultdict(int)  # Tracks number of scans with each class
    label_counts = defaultdict(list)   # Stores pixel counts of each label per scan
    
    # Loop through each segmentation file in the dataset
    for img_dir in image_dir:
        seg_path = glob.glob(img_dir + '/*_seg.nii')
        if not seg_path:
            print(f"No segmentation file found in {img_dir}")
            continue

        # Load segmentation data
        seg_data = nib.load(seg_path[0]).get_fdata()
        unique_labels, counts = np.unique(seg_data, return_counts=True)
        
        # Track presence of each label in this scan
        for label in unique_labels:
            class_presence[label] += 1

        # Store pixel counts for each label in this scan
        for label, count in zip(unique_labels, counts):
            label_counts[label].append(count)
            
    # Print results for class variability
    print("\nClass Variability:")
    for label, count in class_presence.items():
        print(f"Label {int(label)} is present in {count} out of {len(image_dir)} scans ({(count / len(image_dir)) * 100:.2f}%)")

    # Calculate class imbalance
    print("\nClass Imbalance:")
    for label, counts in label_counts.items():
        total_count = sum(counts)
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        print(f"Label {int(label)} - Total pixels: {total_count}, Mean per scan: {mean_count:.2f}, Std Dev: {std_count:.2f}")
    
    # Visualize the average pixel count per label
    labels = list(label_counts.keys())
    avg_pixel_counts = [np.mean(label_counts[label]) for label in labels]
    
    # Adjust the value of the '0' label
    if 0 in labels:
        max_other_counts = max([count for label, count in zip(labels, avg_pixel_counts) if label != 0])
        zero_label_index = labels.index(0)
        avg_pixel_counts[zero_label_index] = max_other_counts * 1.1  # Set '0' slightly above the max other label

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(labels, avg_pixel_counts, color='skyblue')
    plt.xlabel("Label")
    plt.ylabel("Average Pixel Count per Scan")
    plt.title("Average Pixel Count for Each Label Across Scans")

    # Custom tick for label '0'
    if 0 in labels:
        plt.xticks(ticks=labels, labels=[f"> {int(max_other_counts)}" if label == 0 else str(label) for label in labels])

    plt.show()

#Analyze the training data files
def analyze_nifti_file(file_path):
    # Load the file
    img = nib.load(file_path)
    data = img.get_fdata()
    
    # Print basic information
    print(f"\nFile: {file_path}")
    print("Data shape:", data.shape)
    print("Data type:", data.dtype)
    print("Unique values (first 10):", np.unique(data)[:10])
    
    # Display the central slice for quick visual inspection
    plt.imshow(data[data.shape[0] // 2, :, :], cmap="gray")
    plt.title(os.path.basename(file_path))
    plt.axis("off")
    plt.show()

def analyze_random_subjects(data_path, num_subjects=1):
    # List all subdirectories (subjects) in the data path
    subjects = [os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    # Randomly select a specified number of subjects
    random_subjects = random.sample(subjects, num_subjects)
    
    for subject_folder in random_subjects:
        print(f"\nAnalyzing subject folder: {subject_folder}")
        
        # Analyze each file in the subject folder
        for file_name in os.listdir(subject_folder):
            if file_name.endswith(".nii"):
                file_path = os.path.join(subject_folder, file_name)
                analyze_nifti_file(file_path)



def analyse_initial_data():
    """
    Function to visualize different MRI sequences and the mask.
    Objective: To visually inspect sample images from different MRI sequences
    to understand their intensity distribution and alignment.
    """
    fig, (axis1, axis2, axis3, axis4, axis5) = plt.subplots(1, 5, figsize=(20, 10))
    slice_w = 25
    axis1.imshow(VIEW_IMAGE['flair'][:, :, VIEW_IMAGE['flair'].shape[0]//2 - slice_w], cmap='gray')
    axis1.set_title('Flair')
    axis2.imshow(VIEW_IMAGE['t1'][:, :, VIEW_IMAGE['t1'].shape[0]//2 - slice_w], cmap='gray')
    axis2.set_title('T1')
    axis3.imshow(VIEW_IMAGE['t1ce'][:, :, VIEW_IMAGE['t1ce'].shape[0]//2 - slice_w], cmap='gray')
    axis3.set_title('T1CE')
    axis4.imshow(VIEW_IMAGE['t2'][:, :, VIEW_IMAGE['t2'].shape[0]//2 - slice_w], cmap='gray')
    axis4.set_title('T2')
    axis5.imshow(VIEW_IMAGE['mask'][:, :, VIEW_IMAGE['mask'].shape[0]//2 - slice_w])
    axis5.set_title('Mask')
    plt.show()

def check_image_shapes(image_dir):
    """
    Function to check the shape consistency of images in the dataset.
    Objective: Ensure all images have consistent dimensions across all sequences,
    as differing shapes can complicate training.
    """
    shapes = [nib.load(glob.glob(img_dir + '/*')[0]).shape for img_dir in image_dir]
    unique_shapes = set(shapes)
    print("Unique image shapes:", unique_shapes)

def check_intensity_range(image_dir):
    """
    Function to calculate and visualize intensity range for images in each MRI sequence.
    Objective: Identify intensity scaling issues, as large variations may require normalization
    or standardization for consistent model training.
    """
    for modality in ['flair', 't1', 't1ce', 't2']:
        intensities = []
        for img_dir in image_dir:
            img_path = glob.glob(img_dir + f'\\*{modality}*.nii')[0]
            img_data = nib.load(img_path).get_fdata()
            intensities.append((img_data.min(), img_data.max()))
        min_vals, max_vals = zip(*intensities)
        print(f"Modality: {modality}, Min Intensity Range: {min(min_vals)} - {max(max_vals)}")

def plot_random_slices(image_dir):
    """
    Function to plot random slices from the training dataset.
    Objective: Verify consistency of tumor localization across MRI sequences
    and masks by overlaying slices from each modality.
    """
    idx = random.choice(range(len(image_dir)))
    img_paths = sorted(glob.glob(image_dir[idx] + '\\*'))
    
    # Load each modality's data and retrieve the slice at the middle of the third dimension
    slices = [nib.load(img_path).get_fdata()[:, :, nib.load(img_paths[0]).get_fdata().shape[2] // 2] for img_path in img_paths[:4]]
    
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    for i, (slice_data, ax) in enumerate(zip(slices, axes)):
        ax.imshow(slice_data, cmap='gray')
        ax.set_title(['Flair', 'T1', 'T1CE', 'T2'][i])
    plt.show()


def check_segmentation_distribution(image_dir):
    """
    Function to analyze the distribution of labels within segmentation files (_seg.nii).
    Objective: Understand class imbalance in segmentation labels, as this can impact model performance.
    """
    seg_pixel_counts = []
    
    for img_dir in image_dir:
        # Look for segmentation files with _seg.nii in their names
        seg_paths = glob.glob(os.path.join(img_dir, '*_seg.nii'))
        if not seg_paths:
            print(f"No segmentation file found for directory: {img_dir}")
            continue  # Skip this directory if no segmentation file is found

        # Load and analyze the segmentation file
        seg_data = nib.load(seg_paths[0]).get_fdata()
        
        # Calculate the pixel distribution and append to list
        seg_pixel_counts.append(np.bincount(seg_data.astype(int).flatten()))

    # Aggregate results if data was collected
    if seg_pixel_counts:  
        seg_pixel_counts = np.array(seg_pixel_counts)
        print("Mean pixel distribution across segmentation files:", np.mean(seg_pixel_counts, axis=0))
        print("Std deviation in pixel distribution:", np.std(seg_pixel_counts, axis=0))
    else:
        print("No valid segmentation data found in the dataset.")



def visualize_intensity_histograms(image_dirs):
    """
    Function to plot histograms of pixel intensities for each MRI sequence
    (flair, t1, t1ce, t2) across multiple images in one figure.
    Objective: Identify intensity scaling differences across images, which
    can help decide if per-image normalization is necessary.
    """
    modalities = ['flair', 't1', 't1ce', 't2']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 2x2 subplot for 4 modalities
    axes = axes.ravel()  # Flatten to easily index subplots

    for idx, modality in enumerate(modalities):
        ax = axes[idx]
        for img_dir in random.sample(image_dirs, 5):  # Randomly select 5 directories
            img_path = glob.glob(img_dir + f'/*{modality}*.nii')[0]
            img_data = nib.load(img_path).get_fdata().flatten()
            # Use f-string correctly to add filename as label
            ax.hist(img_data, bins=500, alpha=0.5, label=img_dir.split("\\")[-1], histtype='step', linewidth=1.2)

        ax.set_title(f'Intensity Histogram for {modality.upper()}')
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Frequency (log scale)')
        ax.set_yscale('log')  # Log scale for better visualization
        ax.legend(loc='upper right', fontsize='small', frameon=False)

    plt.tight_layout()  # Adjust layout so labels do not overlap
    plt.show()

# Define the path to the training data directory
training_data_path = r"data\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
training_data_directory = "data\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\*"
analyze_random_subjects(training_data_path, num_subjects=1)

IMAGE_DIR = glob.glob(training_data_directory)
CSV_LIST = glob.glob(training_data_directory + 'csv')

for i in CSV_LIST:
    IMAGE_DIR.remove(i)


# Limit to 100 samples for testing
IMAGE_DIR = IMAGE_DIR[:100]
idx = random.choice(range(len(IMAGE_DIR)))

TRAIN_LIST, VAL_LIST = train_test_split(IMAGE_DIR, test_size=0.2)
TRAIN_LIST, TEST_LIST = train_test_split(TRAIN_LIST, test_size=0.3)

# Load and visualize one image sample
VIEW_IMG_IDX = random.randint(0, len(TRAIN_LIST) - 1)
LIST_DATA = sorted(glob.glob(TRAIN_LIST[VIEW_IMG_IDX] + '/*'))
VIEW_IMAGE = {
    'flair': nib.load(LIST_DATA[0]).get_fdata(),
    't1': nib.load(LIST_DATA[2]).get_fdata(),
    't1ce': nib.load(LIST_DATA[3]).get_fdata(),
    't2': nib.load(LIST_DATA[4]).get_fdata(),
    'mask': nib.load(LIST_DATA[1]).get_fdata()
}


# Data normalization for the sample flair image
test_image_flair = nib.load("data\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_355\BraTS20_Training_355_flair.nii").get_fdata()
scaler = MinMaxScaler()
test_image_flair = scaler.fit_transform(test_image_flair.reshape(-1, test_image_flair.shape[-1])).reshape(test_image_flair.shape)

# Define the path to the training data directory and get a list of subject directories
subject_dirs = [os.path.join(training_data_path, d) for d in os.listdir(training_data_path) if os.path.isdir(os.path.join(training_data_path, d))]

# Run all analyses
analyse_initial_data()
check_image_shapes(IMAGE_DIR)
check_intensity_range(IMAGE_DIR)
plot_random_slices(IMAGE_DIR)
check_segmentation_distribution(IMAGE_DIR)
visualize_intensity_histograms(IMAGE_DIR)
# Run the analysis
analyze_class_distribution(subject_dirs)
