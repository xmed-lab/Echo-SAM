import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.colors as mcolors

# ==========================================
# 1. 数据定义
# ==========================================

mapping = {
    "left_ventricle":   'LV',  
    "right_ventricle":  'RV',  
    "left_atrium":      'LA',  
    "right_atrium":     'RA',  
    "interventricular_septum": 'IVS', 
    "interatrial_septum":      'IAS', 
    "mitral_valve":     'MR',  
    "tricuspid_valve":  'TRV',   
    "myocardium":       'Myo',   
}

structure_graph = {
    "left_ventricle": ["left_atrium", "interventricular_septum", "mitral_valve", "myocardium"],
    "right_ventricle": ["right_atrium", "interventricular_septum", "tricuspid_valve", "myocardium"],
    "left_atrium": ["left_ventricle", "interatrial_septum", "mitral_valve"],
    "right_atrium": ["right_ventricle", "interatrial_septum", "tricuspid_valve"],
    "interventricular_septum": ["left_ventricle", "right_ventricle", "myocardium"],
    "interatrial_septum": ["left_atrium", "right_atrium"],
    "mitral_valve": ["left_atrium", "left_ventricle"],
    "tricuspid_valve": ["right_atrium", "right_ventricle"],
    "myocardium": ["left_ventricle"]
}

node_categories = {
    "chambers": ["left_ventricle", "right_ventricle", "left_atrium", "right_atrium"],
    "valves": ["mitral_valve", "tricuspid_valve"],
    "muscle_structures": ["interventricular_septum", "interatrial_septum"],
    "myocardium_tissue": ["myocardium"]
}

color_map_dict = {
    "chambers": "#FF6B6B",       
    "valves": "#04FF8AC8",         
    "muscle_structures": "#048EFFFF", 
    "myocardium_tissue": "#FFAE00" 
}

# ==========================================
# 2. 手动布局
# ==========================================
manual_positions = {
    "left_ventricle":   (-4.2, -3.5, 1.0),  
    "right_ventricle":  ( 4.2, -5.5, 1.8),  
    "left_atrium":      (-4.2,  1.5, 1.0),  
    "right_atrium":     ( 4.2,  1.5, 1.8),  
    "interventricular_septum": (0.0, -1.0, 0.4), 
    "interatrial_septum":      (0.0,  2.5, 0.4), 
    "mitral_valve":     (-3.2,  0.0, 0.0),  
    "tricuspid_valve":  ( 3.2, -0.5, 0.8),   
    "myocardium":       (-4.2, -3.5, -0.5)   
}

# 主簇配置
spread_radius_map = {
    "chambers": 1.2,          
    "valves": 1.9,            
    "muscle_structures": 1.2,
    "myocardium_tissue": 1.5 
}

points_per_node_map = {
    "chambers": 300,          
    "valves": 120, 
    "muscle_structures": 120,
    "myocardium_tissue": 160
}

# ==========================================
# 3. 【新增】噪点配置 (Noise Configuration)
# ==========================================
# 在这里定义每个类别生成的噪点数量和扩散程度
# count: 噪点个数
# spread_factor: 噪点分布的标准差是主簇半径的多少倍 (越大越分散)
noise_config = {
    "chambers": {
        "count": 20,          # 心腔周围加 150 个噪点
        "spread_factor": 2.5   # 分布在半径 2.5 倍远的地方
    },
    "valves": {
        "count": 180,
        "spread_factor": 4.0   # 瓣膜周围的噪点飞得更远
    },
    "muscle_structures": {
        "count": 200,
        "spread_factor": 4.0
    },
    "myocardium_tissue": {
        "count": 200,          # 心肌周围加较多噪点
        "spread_factor": 2.2
    }
}

# ==========================================
# 4. 构建图与生成点云 (含噪点)
# ==========================================
G = nx.Graph()
for node, neighbors in structure_graph.items():
    category = "other"
    for cat, nodes_in_cat in node_categories.items():
        if node in nodes_in_cat:
            category = cat
            break
    G.add_node(node, category=category)
    for neighbor in neighbors:
        if neighbor in structure_graph:
            G.add_edge(node, neighbor)

center_positions = manual_positions
all_x, all_y, all_z = [], [], []
all_colors_rgba = []

light_vec = np.array([0, 1, 1]) 
light_vec = light_vec / np.linalg.norm(light_vec)

for node in G.nodes():
    if node not in center_positions:
        continue
    
    cx, cy, cz = center_positions[node]
    cat = G.nodes[node]['category']
    
    if cat not in color_map_dict:
        continue
        
    hex_color = color_map_dict[cat]
    rgb = mcolors.to_rgb(hex_color)
    
    # --- A. 生成主簇点云 (Core Points) ---
    n_points = points_per_node_map.get(cat, 200)
    radius = spread_radius_map.get(cat, 0.5)
    scale = radius / 2.2
    
    xs_core = np.random.normal(loc=cx, scale=scale, size=n_points)
    ys_core = np.random.normal(loc=cy, scale=scale, size=n_points)
    zs_core = np.random.normal(loc=cz, scale=scale, size=n_points)
    
    # 主簇光影计算
    vectors_core = np.vstack((xs_core - cx, ys_core - cy, zs_core - cz)).T
    norms_core = np.linalg.norm(vectors_core, axis=1, keepdims=True)
    norms_core[norms_core == 0] = 1e-6
    normals_core = vectors_core / norms_core
    
    intensity_core = np.dot(normals_core, light_vec) * 0.5 + 0.5
    intensity_core = np.clip(intensity_core, 0.4, 1.0)
    
    for i in range(n_points):
        r, g, b = rgb
        lit_r = r * intensity_core[i]
        lit_g = g * intensity_core[i]
        lit_b = b * intensity_core[i]
        
        depth_alpha = 0.85 + (zs_core[i]) * 0.05 
        depth_alpha = np.clip(depth_alpha, 0.4, 0.95)
        
        all_x.append(xs_core[i])
        all_y.append(ys_core[i])
        all_z.append(zs_core[i])
        all_colors_rgba.append((lit_r, lit_g, lit_b, depth_alpha))

    # --- B. 生成噪点 (Noise Points) - 非高斯、不均匀分布 ---
    noise_settings = noise_config.get(cat, {"count": 0, "spread_factor": 0})
    n_noise = noise_settings["count"]
    factor = noise_settings["spread_factor"]
    
    if n_noise > 0:
        # 基础半径范围
        max_radius = (radius / 2.2) * factor
        
        # 1. 生成均匀分布的随机方向 (球坐标采样)
        # u, v 是 [0, 1] 之间的均匀随机数
        u = np.random.uniform(0, 1, n_noise)
        v = np.random.uniform(0, 1, n_noise)
        
        # 转换为球坐标角度
        theta = 2 * np.pi * u             # 方位角 [0, 2π]
        phi = np.arccos(2 * v - 1)        # 极角 [0, π]，保证球面上均匀分布
        
        # 2. 生成不均匀的半径距离
        # 方案 A: 完全均匀 (Uniform) -> r = R * cubic_root(random)
        # 方案 B: 边缘偏好 (Edge biased) -> 让点更倾向于分布在外部，模拟扩散
        # 这里我们使用 r = R * (random ^ 0.3)，指数小于1会让点更偏向外部边缘
        random_radii = max_radius * (np.random.uniform(0, 1, n_noise) ** 0.3)
        
        # 3. 将球坐标转换为笛卡尔坐标 (x, y, z)
        dx = random_radii * np.sin(phi) * np.cos(theta)
        dy = random_radii * np.sin(phi) * np.sin(theta)
        dz = random_radii * np.cos(phi)
        
        # 4. 【可选】添加方向性拉伸 (模拟不均匀伪影)
        # 例如：让噪点在 Z 轴方向拉长得更多，模拟声束方向的拖尾
        # 修改下面的系数可以改变形状 (1.0=球形, >1.0=橄榄形/长条形)
        stretch_x = np.random.rand()*1  # Z轴拉长 1.5 倍
        stretch_z = np.random.rand()*2  # Z轴拉长 1.5 倍
        stretch_y = np.random.rand()*2.5 # Y轴轻微拉长
        
        dx = dx * stretch_x
        dy = dy * stretch_y
        dz = dz * stretch_z
        
        # 最终坐标
        xs_noise = cx + dx
        ys_noise = cy + dy
        zs_noise = cz + dz
        
        for i in range(n_noise):
            r, g, b = rgb
            # 噪点颜色：稍微去色并调暗，模拟低信噪比
            lit_r = r * 0.6
            lit_g = g * 0.6
            lit_b = b * 0.6
            
            # 透明度：距离中心越远，透明度越低 (模拟信号衰减)
            dist_from_center = np.sqrt(dx[i]**2 + dy[i]**2 + dz[i]**2)
            # 归一化距离用于计算 alpha
            norm_dist = dist_from_center / max_radius if max_radius > 0 else 0
            base_alpha = 0.4
            fade_alpha = base_alpha * (1.0 - min(norm_dist, 1.0) * 0.6) # 越远越透明
            
            depth_alpha = fade_alpha + (zs_noise[i]) * 0.02 
            depth_alpha = np.clip(depth_alpha, 0.05, 0.5) 
            
            all_x.append(xs_noise[i])
            all_y.append(ys_noise[i])
            all_z.append(zs_noise[i])
            all_colors_rgba.append((lit_r, lit_g, lit_b, depth_alpha))

# 转换为 numpy 数组
all_x = np.array(all_x)
all_y = np.array(all_y)
all_z = np.array(all_z)

# ==========================================
# 5. 绘图
# ==========================================
fig = plt.figure(figsize=(14, 14)) 
ax = fig.add_subplot(111, projection='3d')

# A. 连线 (只连主中心，不连噪点)
for edge in G.edges():
    if edge[0] in center_positions and edge[1] in center_positions:
        p0 = center_positions[edge[0]]
        p1 = center_positions[edge[1]]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], 
                color='#555555', alpha=0.5, linewidth=3.2, linestyle='--')

# B. 点云 (包含主簇和噪点)
colors_array = np.array(all_colors_rgba)
sc = ax.scatter(
    all_x, all_y, all_z,
    c=colors_array,
    s=40,             # 噪点较多时，适当减小点尺寸以免糊成一团
    edgecolors='none',
    depthshade=True,
    rasterized=False
)

# C. 标签 (仅在主簇中心)
for node in G.nodes():
    if node not in center_positions:
        continue
    
    cx, cy, cz = center_positions[node]
    # label = node.replace('_', ' ').title()
    label = mapping[node]
    
    ax.text(cx, cy, cz, 
            label, 
            fontsize=13,          
            ha='center', 
            va='center',
            fontweight='bold',
            color='#FFFFFF',      
            bbox=dict(
                facecolor='black', 
                alpha=0.7,         # 稍微增加背景不透明度以穿透噪点
                edgecolor='none', 
                pad=0.3,           
                boxstyle='round,pad=0.3'
            ),
            zorder=20             
            )

# ==========================================
# 6. 自适应缩放 (包含噪点范围)
# ==========================================
ax.axis('off')
ax.view_init(elev=25, azim=-60)

ax.set_facecolor('#FFFFFF')
fig.patch.set_facecolor('#FFFFFF')

# 计算包含噪点在内的实际边界
margin_ratio = 0.10  # 噪点已经很远了，边距可以稍微小一点 (10%)
x_range = all_x.max() - all_x.min()
y_range = all_y.max() - all_y.min()
z_range = all_z.max() - all_z.min()

x_pad = x_range * margin_ratio
y_pad = y_range * margin_ratio
z_pad = z_range * margin_ratio

ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
ax.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)
ax.set_zlim(all_z.min() - z_pad, all_z.max() + z_pad)

try:
    ax.set_box_aspect([x_range, y_range, z_range]) 
except AttributeError:
    pass

# 保存
output_pdf = "echo_4chamber_with_noise.pdf"
output_png = "echo_4chamber_with_noise.png"

print("正在渲染带噪点的四腔心图像...")
print(f"总点数：{len(all_x)} (其中主簇约 {sum(points_per_node_map.values())}, 噪点约 {sum(c['count'] for c in noise_config.values())})")
plt.savefig(output_pdf, format='pdf', bbox_inches='tight', dpi=300)
print(f"✅ 成功保存矢量图: {output_pdf}")

plt.savefig(output_png, format='png', bbox_inches='tight', dpi=400)
print(f"✅ 成功保存高清预览: {output_png}")

plt.show()