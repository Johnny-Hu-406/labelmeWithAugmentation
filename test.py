import os

def get_all_images(root_folder):
    image_list = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_list.append(os.path.join(subdir, file))
    return sorted(image_list)

root_folder = "input_data"  # 設定主資料夾
image_list = get_all_images(root_folder)

# 印出所有圖片的完整路徑
for img in image_list:
    print(img)
