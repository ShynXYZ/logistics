# ============================================
# 🚚 Транспорттық компания деректерін K-Means және иерархиялық кластерлеу
# ============================================

# Қажетті кітапханалар
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# --------------------------------------------
# 1. Деректерді жасау және жүктеу
# --------------------------------------------
np.random.seed(42)

# Үш кластер үшін параметрлер
clusters = {
    'small': {'size': 100, 'distance_range': (50, 200), 'vehicles': (1, 4)},
    'medium': {'size': 100, 'distance_range': (200, 400), 'vehicles': (4, 8)},
    'large': {'size': 100, 'distance_range': (400, 600), 'vehicles': (8, 15)}
}

data = []
for cluster, params in clusters.items():
    for i in range(params['size']):
        distance = np.random.randint(params['distance_range'][0], params['distance_range'][1])
        delivery_time = int(distance * 0.15 + np.random.normal(10, 5))
        vehicles = np.random.randint(params['vehicles'][0], params['vehicles'][1])
        employees = vehicles + np.random.randint(2, 5)
        volume = int(distance * 0.4 + np.random.normal(20, 10))
        fuel = int(distance * 0.15 + np.random.normal(5, 3))
        maintenance = int(vehicles * 200 + np.random.normal(100, 50))
        revenue = int(distance * 18 + np.random.normal(500, 200))
        
        on_time_rate = max(70, min(99, 100 - distance * 0.03 + np.random.normal(5, 3)))
        satisfaction = max(65, min(98, on_time_rate - np.random.normal(5, 2)))
        
        data.append([
            distance, delivery_time, vehicles, employees, volume, 
            fuel, maintenance, revenue, satisfaction, on_time_rate
        ])

# DataFrame жасау
columns = [
    'distance_km', 'delivery_time_hours', 'vehicle_count', 'employee_count',
    'package_volume_m3', 'fuel_consumption_l', 'maintenance_cost', 
    'freight_revenue', 'customer_satisfaction', 'on_time_delivery_rate'
]

data = pd.DataFrame(data, columns=columns)
print("🔹 Алғашқы 5 жол:\n", data.head())

# --------------------------------------------
# 2. Деректерді масштабтау
# --------------------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print(f"\n🔹 Масштабталған деректер өлшемі: {scaled_data.shape}")

# --------------------------------------------
# 3. K-Means кластеризациясы
# --------------------------------------------
k_means = KMeans(init='k-means++', n_clusters=3, n_init=15, random_state=42)
k_means.fit(scaled_data)

k_means_cluster_centers = k_means.cluster_centers_
k_means_labels = pairwise_distances_argmin(scaled_data, k_means_cluster_centers)

print("\n🔹 Кластер орталықтары:\n", k_means_cluster_centers)

# --------------------------------------------
# 4. Нәтижелерді визуализациялау (жұптастыру түрінде)
# --------------------------------------------
f, ax = plt.subplots(1, 3, figsize=(18, 5))
colors = ['#4EACC5', '#FF9C34', '#4E9A06']

# Кластерлерді визуализациялау әртүрлі комбинацияларда
feature_pairs = [
    (0, 1),   # distance_km vs delivery_time_hours
    (4, 5),   # package_volume_m3 vs fuel_consumption_l
    (7, 9)    # freight_revenue vs on_time_delivery_rate
]

feature_names = [
    ('Қашықтық (км)', 'Жеткізу уақыты (сағ)'),
    ('Жүк көлемі (м³)', 'Отын шығыны (л)'),
    ('Табыс', 'Уақыттылық (%)')
]

for i in range(3):
    my_members = k_means_labels == i
    cluster_center = k_means_cluster_centers[i]
    
    for j, (idx1, idx2) in enumerate(feature_pairs):
        ax[j].scatter(scaled_data[my_members, idx1], scaled_data[my_members, idx2], 
                     c=colors[i], alpha=0.7, s=40, label=f'Кластер {i+1}')
        ax[j].scatter(cluster_center[idx1], cluster_center[idx2], 
                     c=colors[i], marker='x', s=200, linewidth=3, label=f'Орталық {i+1}')
        
        ax[j].set_xlabel(feature_names[j][0])
        ax[j].set_ylabel(feature_names[j][1])
        ax[j].legend()
        ax[j].grid(True, alpha=0.3)

plt.suptitle('🚚 K-Means кластерлеу нәтижелері (транспорттық компания)')
plt.tight_layout()
plt.show()

# --------------------------------------------
# 5. Шынтақ (локтя) әдісі арқылы кластерлер санын анықтау
# --------------------------------------------
inertia_values = []

for n_clusters in range(1, 11):
    k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=15, random_state=42)
    k_means.fit(scaled_data)
    inertia_values.append(k_means.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia_values, marker='o', linewidth=2, markersize=8)
plt.title('📈 Шынтақ әдісі - Оңтайлы кластер санын анықтау')
plt.xlabel('Кластер саны')
plt.ylabel('Инерция')
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 11))
plt.show()

# --------------------------------------------
# 6. Иерархиялық кластеризация
# --------------------------------------------
hier = AgglomerativeClustering(n_clusters=3, linkage='ward')
hier_labels = hier.fit_predict(scaled_data)

# Визуализация
f, ax = plt.subplots(1, 3, figsize=(18, 5))

for i in range(3):
    my_members = hier_labels == i
    
    for j, (idx1, idx2) in enumerate(feature_pairs):
        ax[j].scatter(scaled_data[my_members, idx1], scaled_data[my_members, idx2], 
                     c=colors[i], alpha=0.7, s=40, label=f'Кластер {i+1}')
        
        ax[j].set_xlabel(feature_names[j][0])
        ax[j].set_ylabel(feature_names[j][1])
        ax[j].legend()
        ax[j].grid(True, alpha=0.3)

plt.suptitle('🌿 Иерархиялық кластерлеу нәтижелері (транспорттық компания)')
plt.tight_layout()
plt.show()

# --------------------------------------------
# 7. Дендограмма құру
# --------------------------------------------
Z = linkage(scaled_data, method='ward')
plt.figure(figsize=(12, 8))
dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=45, leaf_font_size=10)
plt.title('🌳 Дендограмма - Иерархиялық кластерлеу')
plt.xlabel('Дерек нүктелерінің индексі')
plt.ylabel('Қашықтық')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------
# 8. Нәтижелерді салыстыру және талдау
# --------------------------------------------
print("\n" + "="*50)
print("📊 КЛАСТЕРЛЕУ НӘТИЖЕЛЕРІН ТАЛДАУ")
print("="*50)

# Кластерлер бойынша статистика
data['kmeans_cluster'] = k_means_labels
data['hierarchical_cluster'] = hier_labels

print("\n🔹 K-Means кластерлері бойынша орташа мәндер:")
kmeans_stats = data.groupby('kmeans_cluster').agg({
    'distance_km': 'mean',
    'freight_revenue': 'mean', 
    'customer_satisfaction': 'mean',
    'vehicle_count': 'mean'
}).round(2)

print(kmeans_stats)

print("\n🔹 Иерархиялық кластерлер бойынша орташа мәндер:")
hier_stats = data.groupby('hierarchical_cluster').agg({
    'distance_km': 'mean',
    'freight_revenue': 'mean',
    'customer_satisfaction': 'mean', 
    'vehicle_count': 'mean'
}).round(2)

print(hier_stats)

# --------------------------------------------
# 9. Кластерлерді сипаттау
# --------------------------------------------
print("\n" + "="*50)
print("🎯 КЛАСТЕРЛЕРДІҢ СИПАТТАМАСЫ")
print("="*50)

cluster_descriptions = {
    0: "🚛 Үлкен тасымалдаулар - Ұзын қашықтық, жоғары табыс, төмен қанағаттану",
    1: "🚚 Орташа тасымалдаулар - Орташа қашықтық, тұрақты табыс", 
    2: "🚗 Кіші тасымалдаулар - Қысқа қашықтық, төмен табыс, жоғары қанағаттану"
}

for cluster_num in range(3):
    cluster_data = data[data['kmeans_cluster'] == cluster_num]
    print(f"\n🔸 {cluster_descriptions[cluster_num]}:")
    print(f"   - Жазбалар саны: {len(cluster_data)}")
    print(f"   - Орташа қашықтық: {cluster_data['distance_km'].mean():.1f} км")
    print(f"   - Орташа табыс: {cluster_data['freight_revenue'].mean():.0f}")
    print(f"   - Орташа қанағаттану: {cluster_data['customer_satisfaction'].mean():.1f}%")
    print(f"   - Орташа көлік саны: {cluster_data['vehicle_count'].mean():.1f}")

print("\n" + "="*50)
print("✅ КЛАСТЕРЛЕУ АЯҚТАЛДЫ!")
print("="*50)
