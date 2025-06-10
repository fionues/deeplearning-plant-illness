
# Configuration parameters for plant disease detection

# Path to the CSV file
CSV_FILE = "../data/image_data_rel.csv"

# Original and target image sizes
ORIGINAL_IMG_SIZE = (256, 256)  # Original image size before processing
TARGET_IMG_SIZE = (299, 299)    # Target image size (299 required by InceptionV3)

# Dataset settings
IMAGES_PER_LABEL = 200          # Number of images per label (max ca 200 - 1000)
VALIDATION_SPLIT = 0.2         # Fraction of data for validation

# Training settings
EPOCHS = 20                    # Number of training epochs
BATCH_SIZE = 20                # Batch size for training
