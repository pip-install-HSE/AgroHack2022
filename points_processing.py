import numpy as np
import open3d as o3d
import json
import cv2

# Read .ply file
with open(input("Enter metadata file path: "), "r") as f:
    metadata = json.loads(f.read())
    
input_file = input("Enter pointcloud ply file path: ")
pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud
# o3d.visualization.draw_geometries([pcd]) 

intr = o3d.camera.PinholeCameraIntrinsic(1024, 768, metadata["intrinsics"]["fx"], metadata["intrinsics"]["fy"], metadata["intrinsics"]["ppx"], metadata["intrinsics"]["ppy"])
cam = o3d.camera.PinholeCameraParameters()
cam.intrinsic = intr

# renderer.setup_camera(cam.intrinsic, cam.extrinsic)
# depth_img = renderer.render_to_depth_image()
# o3d.io.write_image("depth.jpg", depth_img)
# control = get_view_control()
# control.convert_from_pinhole_camera_parameters(cam, True)
#yaw pitch roll - look from above 0, 500

#pitch yaw roll
#pitch roll yaw
#roll pitch yaw
#roll yaw pitch
#yaw roll pitch
R = pcd.get_rotation_matrix_from_xyz((np.pi,0,0))
# R = pcd.get_rotation_matrix_from_xyz((metadata['yaw_rad'], metadata['pitch_rad'], metadata['roll_rad']))
cam.extrinsic = np.array([[*R[0], 0], [*R[1], 0], [*R[2], 0.], [0., 0., 0., 1.]])
pcd.rotate(R, center = (0,0,0))

# Visualize the point cloud within open3d
# o3d.visualization.draw_geometries([pcd]) 
render = o3d.visualization.Visualizer()
render.create_window()
# model, mat = getModel()
control = render.get_view_control()
render.add_geometry(pcd) 
control.convert_from_pinhole_camera_parameters(cam, True)
render.update_geometry(pcd) 
# render.setup_camera(cam.intrinsic, cam.extrinsic)
render.run()

# offscreen = o3d.visualization.rendering.OffscreenRenderer(1920, 1080)
# depth_img = offscreen.render_to_image()
# print(depth_img)
# o3d.io.write_image("depth.jpg", depth_img, quality=100)
# depth_img = renderer.render_to_depth_image()
# o3d.visualization.draw_geometries([depth_img])

# Convert open3d format to numpy array
# Here, you have the point cloud in numpy format. 
point_cloud_in_numpy = np.asarray(pcd.points) 