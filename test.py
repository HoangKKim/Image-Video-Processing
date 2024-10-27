import numpy as np
import cv2

def region_growing(image, seed, threshold):
    rows, cols = image.shape
    visited = np.zeros_like(image)  # Mảng để theo dõi các điểm đã thăm
    region = np.zeros_like(image)   # Mảng để lưu kết quả khu vực

    # Hàm kiểm tra điều kiện mở rộng
    def check_condition(pixel_value, seed_value):
        return abs(pixel_value - seed_value) < threshold

    # Hàm kiểm tra sự đồng nhất của khu vực
    def check_homogeneity(region, seed_value):
        return np.all(np.abs(region - seed_value) < threshold)

    def grow_region(x, y, seed_value):
        # Kiểm tra xem điểm có nằm trong hình ảnh không
        if 0 <= x < rows and 0 <= y < cols:
            # Kiểm tra xem điểm đã được thăm hay chưa
            if not visited[x, y]:
                # Kiểm tra điều kiện mở rộng
                if check_condition(image[x, y], seed_value):
                    # Thêm điểm vào khu vực và đánh dấu đã thăm
                    region[x, y] = image[x, y]
                    visited[x, y] = 1

                    # Kiểm tra sự đồng nhất của khu vực sau mỗi bước mở rộng
                    if check_homogeneity(region, seed_value):
                        # Tiếp tục mở rộng theo 8 hướng lân cận
                        grow_region(x + 1, y, seed_value)
                        grow_region(x - 1, y, seed_value)
                        grow_region(x, y + 1, seed_value)
                        grow_region(x, y - 1, seed_value)
                        grow_region(x + 1, y + 1, seed_value)  # Hướng chéo
                        grow_region(x - 1, y + 1, seed_value)  # Hướng chéo
                        grow_region(x + 1, y - 1, seed_value)  # Hướng chéo
                        grow_region(x - 1, y - 1, seed_value)  # Hướng chéo

    # Bắt đầu mở rộng từ hạt giống
    seed_value = image[seed]
    grow_region(*seed, seed_value)

    return region

# Đọc ảnh đầu vào và chuyển đổi thành ảnh grayscale
input_image = cv2.imread('your_image_path')
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Chọn điểm xuất phát và ngưỡng
seed_point = (100, 100)  # Tùy chỉnh điểm xuất phát
threshold_value = 10      # Tùy chỉnh ngưỡng

# Áp dụng phương pháp Region Growing
result_region = region_growing(gray_image, seed_point, threshold_value)

# Hiển thị ảnh gốc và kết quả
cv2.imshow('Original Image', gray_image)
cv2.imshow('Region Growing Result', result_region)
cv2.waitKey(0)
cv2.destroyAllWindows()


# HAC
import numpy as np

def average_linkage(X):
    # Bước 1: Khởi tạo cụm, mỗi điểm là một cụm riêng lẻ
    clusters = [[i] for i in range(len(X))]
    
    while len(clusters) > 1:
        # Bước 2: Tính ma trận khoảng cách
        distances = np.zeros((len(clusters), len(clusters)))
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                avg_dist = np.mean([np.linalg.norm(X[p1] - X[p2]) for p1 in clusters[i] for p2 in clusters[j]])
                distances[i, j] = avg_dist
                distances[j, i] = avg_dist
        
        # Bước 3: Gom nhóm cặp gần nhất
        min_indices = np.unravel_index(np.argmin(distances), distances.shape)
        cluster1, cluster2 = clusters[min_indices[0]], clusters[min_indices[1]]
        new_cluster = cluster1 + cluster2
        
        # Bước 4: Cập nhật ma trận khoảng cách
        new_distances = np.zeros((len(clusters) - 1, len(clusters) - 1))
        new_distances[0] = np.mean([distances[min_indices[0]], distances[min_indices[1]]], axis=0)
        new_distances[:, 0] = new_distances[0]
        new_distances[1:] = distances[np.setdiff1d(np.arange(len(clusters)), min_indices, assume_unique=True)][:, np.setdiff1d(np.arange(len(clusters)), min_indices, assume_unique=True)]
        
        # Bước 5: Loại bỏ cụm đã gom nhóm và thêm cụm mới
        clusters.pop(max(min_indices))
        clusters.pop(min(min_indices))
        clusters.append(new_cluster)
        
    return clusters[0]

# Ví dụ sử dụng
data_points = np.array([[1, 2], [2, 3], [5, 8], [7, 6], [9, 10]])
result = average_linkage(data_points)
print("Final Cluster:", result)
