from numpy import load
data = load('/home/vader/AONeuS_implementation/aoneus/data/reduced_baseline_0.6x_rgb/cameras_sphere.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])


import pickle


with open('/home/vader/AONeuS_implementation/aoneus/data/reduced_baseline_0.6x_sonar/Data/000.pkl', 'rb') as f:
    data = pickle.load(f)
print(data)



""" import torch

H = 1080
W = 1920
hfov = 1.4

near = 0
far = 1
ray_n_samples = 3
randomize_points = True
n_selected_px = 3

px = torch.tensor([[10, 20], [30, 40], [50, 60]])  # Example tensor with 3 selected pixels
i = px[:, 0]  # Extracts the first column, resulting in tensor([10, 30, 50])
j = px[:, 1]  # Extracts the second column, resulting in tensor([20, 40, 60])


theta = -hfov / 2 + j * hfov / W



# sample ray from near to far
t_vals = torch.linspace(near, far, ray_n_samples).float().repeat(n_selected_px).reshape(n_selected_px, -1)
t_vals_gap = (far - near) / ray_n_samples

# #t_vals pre-randomized
# print(t_vals)


rnd = -t_vals_gap + torch.rand(n_selected_px, ray_n_samples) * 2 * t_vals_gap
if randomize_points:
    t_vals = torch.clip(t_vals + rnd, min=near, max=far)

# # randomized
# print(t_vals)


""" #generate grid of sample points [i, j, t_vals (depth samples)] 
"""
coords = torch.stack((i.repeat_interleave(ray_n_samples).reshape(n_selected_px, -1), 
                        j.repeat_interleave(ray_n_samples).reshape(n_selected_px, -1), 
                        t_vals), dim = -1)
coords = coords.reshape(-1, 3)

print(coords)

X = torch.tan(theta) * torch.ones(n_selected_px)
Y = torch.zeros(n_selected_px)
Z = -torch.ones(n_selected_px)

print('X:', X)
print('Y:', Y)
print('Z:',Z)

dirs = torch.stack((X, Y, Z), dim=-1)
dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)

print(dirs)

origins = torch.zeros((n_selected_px, 3))

print(origins)

dirs = torch.matmul(c2w[:3, :3], dirs.T).T + c2w[:3, 3]
origins = torch.matmul(c2w[:3, :3], origins.T).T + c2w[:3, 3]


samples = origins.unsqueeze(1) + t_vals.unsqueeze(-1) * dirs.unsqueeze(1) """