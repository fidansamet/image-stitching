import os

dataset_name = "HW2_Dataset"
dataset_contents = os.listdir(dataset_name)


def main():
    for content in dataset_contents:
        # If it is a directory get images
        if os.path.isdir(dataset_name + "/" + content):
            print(content)
            subset_images = os.listdir(dataset_name + "/" + content)
            for image_name in subset_images:
                print(image_name)


if __name__ == '__main__':
    main()
