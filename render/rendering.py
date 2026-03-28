"""Offscreen mesh rendering utilities using pyrender.

Provides RenderTool class for rendering SMPLX mesh sequences to video,
with support for optional audio overlay and text annotations.
"""

import os
import shutil
import tempfile
import threading
from subprocess import call

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import cv2
import librosa
import numpy as np
import pyrender
import trimesh
from psbody.mesh import Mesh
from scipy.io import wavfile
from tqdm import tqdm

# Directory of this script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_unit_factor(unit):
    """Return scale factor for converting to the given unit."""
    if unit == 'mm':
        return 1000.0
    elif unit == 'cm':
        return 100.0
    elif unit == 'm':
        return 1.0
    else:
        raise ValueError(f'Unit not supported: {unit}')


def rotate_camera_pose(scene, camera, k):
    """Add camera and lights to the scene with optional rotation.

    Args:
        scene: pyrender Scene instance.
        camera: pyrender Camera instance.
        k: Number of rotation steps (each step rotates 45 degrees around X).

    Returns:
        The scene with camera and lights added.
    """
    camera_pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0.7],
        [0, 0, 1, 1.0],
        [0, 0, 0, 1],
    ])
    theta = np.pi / 4
    R_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1],
    ])

    for _ in range(k):
        camera_pose = R_x @ camera_pose
    camera_pose[:3, 3] += np.array([0, 0, 2])
    scene.add(camera, pose=camera_pose)

    light = pyrender.PointLight(
        color=np.array([1.0, 1.0, 1.0]) * 0.2, intensity=25
    )
    for light_pos in [[0, -2, 2], [0, 2, 2], [1, 1, 2]]:
        light_pose = np.eye(4)
        light_pose[:3, 3] = light_pos
        for _ in range(k):
            light_pose = R_x @ light_pose
        scene.add(light, pose=light_pose)

    return scene


def render_mesh_helper(mesh, t_center, rot=np.zeros(3), tex_img=None,
                       v_colors=None, errors=None, error_unit='m',
                       min_dist_in_mm=0.0, max_dist_in_mm=3.0,
                       xmag=0.55, y=0.7, z=1, camera='o', r=None,
                       rotation_times=0):
    """Render a single mesh frame to an RGBA image.

    Args:
        mesh: psbody Mesh instance with vertices and faces.
        t_center: Center point for the mesh (3,).
        rot: Rotation vector for mesh orientation.
        tex_img: Optional texture image.
        v_colors: Optional per-vertex colors.
        errors: Optional per-vertex error values for heatmap coloring.
        error_unit: Unit of error values ('m', 'cm', or 'mm').
        min_dist_in_mm: Minimum distance for heatmap normalization.
        max_dist_in_mm: Maximum distance for heatmap normalization.
        xmag: Orthographic camera magnification.
        y: Camera Y position.
        z: Camera Z position.
        camera: Camera type: 'o' orthographic, 'i' intrinsic, 'y' perspective.
        r: pyrender OffscreenRenderer instance.
        rotation_times: Number of camera rotation steps.

    Returns:
        RGBA image as numpy array of shape (H, W, 4), uint8.
    """
    camera_params = {
        'c': np.array([0, 0]),
        'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
        'f': np.array([5000, 5000]),
    }
    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = (
        cv2.Rodrigues(rot)[0].dot((mesh_copy.v - t_center).T).T + t_center
    )

    texture_rendering = (
        tex_img is not None and hasattr(mesh, 'vt') and hasattr(mesh, 'ft')
    )

    if texture_rendering:
        tex = pyrender.Texture(source=tex_img, source_channels='RGB')
        material = pyrender.material.MetallicRoughnessMaterial(
            baseColorTexture=tex
        )
        temp_filename = '%s.obj' % next(tempfile._get_candidate_names())
        mesh.write_obj(temp_filename)
        tri_mesh = trimesh.load(temp_filename, process=False)
        try:
            os.remove(temp_filename)
        except OSError:
            print(f'Failed deleting temporary file - {temp_filename}')
        render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material)
    elif errors is not None:
        import matplotlib as mpl
        import matplotlib.cm as cm
        unit_factor = get_unit_factor('mm') / get_unit_factor(error_unit)
        errors = unit_factor * errors
        norm = mpl.colors.Normalize(vmin=min_dist_in_mm, vmax=max_dist_in_mm)
        cmap = cm.get_cmap(name='jet')
        colormapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        rgb_per_v = colormapper.to_rgba(errors)[:, 0:3]
    elif v_colors is not None:
        rgb_per_v = v_colors
    else:
        rgb_per_v = [180, 180, 180, 255]

    color_grey = np.array([0.65, 0.65, 0.65])

    if not texture_rendering:
        tri_mesh = trimesh.Trimesh(
            vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v
        )
        render_mesh = pyrender.Mesh.from_trimesh(
            tri_mesh, smooth=True,
            material=pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.,
                roughnessFactor=1,
                alphaMode='MASK',
                baseColorFactor=(
                    color_grey[0], color_grey[1], color_grey[2], 1.0
                ),
            ),
        )

    scene = pyrender.Scene(
        ambient_light=[.2, .2, .2], bg_color=[0, 0, 0]
    )

    if camera == 'o':
        camera = pyrender.OrthographicCamera(xmag=xmag, ymag=xmag)
    elif camera == 'i':
        camera = pyrender.IntrinsicsCamera(
            fx=camera_params['f'][0], fy=camera_params['f'][1],
            cx=camera_params['c'][0], cy=camera_params['c'][1],
            znear=frustum['near'], zfar=frustum['far'],
        )
    elif camera == 'y':
        camera = pyrender.PerspectiveCamera(yfov=(np.pi / 2.0))

    scene.add(render_mesh, pose=np.eye(4))
    scene = rotate_camera_pose(scene, camera, rotation_times)

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    r.clear_color = np.array([0, 0, 0, 0])
    color, depth = r.render(scene, flags=flags)

    color = color[..., ::-1]
    background_mask = depth == 0
    alpha_channel = np.ones((depth.shape[0], depth.shape[1]))
    alpha_channel[background_mask] = 0
    color_with_alpha = np.dstack(
        [color, alpha_channel * 255]
    ).astype(np.uint8)

    return color_with_alpha


def add_image_text(img, text, color=(0, 0, 255), w=800, h=800):
    """Overlay text and a colored border on an image.

    Args:
        img: Input image array.
        text: Text string to display.
        color: BGR color tuple for text and border.
        w: Image width for border rectangle.
        h: Image height for border rectangle.

    Returns:
        Image with text and border overlay.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = np.require(img, dtype='f4', requirements=['O', 'W'])
    img.flags.writeable = True
    img = img.copy()
    img = cv2.putText(img, '%s' % text, (100, 100), font, 2, color, 2, 1)
    img = cv2.rectangle(img, (0, 0), (w, h), color, thickness=3)
    return img


class Struct(object):
    """Simple attribute container."""

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class RenderTool:
    """Tool for rendering SMPLX mesh sequences to video.

    Uses pyrender offscreen rendering to generate per-frame images,
    then encodes them to video with ffmpeg.

    Args:
        smplx_model_path: Path to the SMPLX_NEUTRAL.npz file.
            If None, uses the default path under render_model/smplx/.
    """

    def __init__(self, smplx_model_path=None):
        if smplx_model_path is None:
            smplx_model_path = os.path.join(
                _SCRIPT_DIR, 'render_model', 'smplx', 'SMPLX_NEUTRAL.npz'
            )
        self.template_mesh = Mesh()
        model_data = np.load(smplx_model_path, allow_pickle=True)
        data_struct = Struct(**model_data)
        self.template_mesh.f = data_struct.f

    def _render_sequences(self, cur_wav_file, v_list, fps=30,
                          video_fname=None, stand=False, face=False,
                          whole_body=False, run_in_parallel=False,
                          multi_view=False, asr_path=None, add_text=False,
                          **kwargs):
        """Render a list of vertex sequences to video.

        Args:
            cur_wav_file: Path to audio file (or None for no audio).
            v_list: List of vertex arrays, each of shape (T, V, 3).
            fps: Video frame rate.
            video_fname: Output video file path.
            stand: Not used (kept for API compatibility).
            face: If True, use close-up face camera settings.
            whole_body: If True, use tall viewport for full body.
            run_in_parallel: If True, run rendering in a thread.
            multi_view: Not used in this simplified version.
            asr_path: Path to ASR data for text overlay.
            add_text: Whether to overlay frame numbers.
        """
        if run_in_parallel:
            thread = threading.Thread(
                target=self._render_sequences_helper,
                args=(video_fname, cur_wav_file, v_list, face, whole_body,
                      add_text, fps, asr_path),
            )
            thread.start()
            thread.join()
        else:
            self._render_sequences_helper(
                video_fname, cur_wav_file, v_list, face, whole_body,
                add_text, fps, asr_path,
            )

    def _render_sequences_helper(self, video_fname, cur_wav_file, v_list,
                                 face=False, whole_body=False,
                                 add_text=True, fps=30, asr_path=None):
        """Core rendering loop: render frames and encode to video.

        Args:
            video_fname: Output video file path.
            cur_wav_file: Path to audio file (or None).
            v_list: List of vertex arrays.
            face: Use face close-up camera.
            whole_body: Use tall viewport.
            add_text: Overlay frame numbers.
            fps: Video frame rate.
            asr_path: Path to ASR data for text overlay.
        """
        num_frames = v_list[0].shape[0]

        # Flip Y and Z axes (dataset convention)
        for v in v_list:
            v = v.reshape(v.shape[0], -1, 3)
            v[:, :, 1] = -v[:, :, 1]
            v[:, :, 2] = -v[:, :, 2]

        viewport_height = 800
        num_video = len(v_list)
        assert num_video in [1, 2, 3, 4], (
            f'Unsupported number of videos: {num_video}'
        )
        if num_video == 1:
            width, height = 800, 800
        elif num_video == 2:
            width, height = 1600, 800
        elif num_video == 3:
            width, height = 2400, 800
        elif num_video == 4:
            width, height = 3200, 800

        if whole_body:
            width, height = 800, 1440
            viewport_height = 1440

        # Prepare audio temp file
        if cur_wav_file is not None:
            audio, sr = librosa.load(cur_wav_file, sr=16000)
            tmp_audio_file = tempfile.NamedTemporaryFile(
                'w', suffix='.wav', dir=os.path.dirname(video_fname)
            )
            tmp_audio_file.close()
            wavfile.write(tmp_audio_file.name, sr, audio)

        # Prepare video temp file and frame output directory
        tmp_video_file = tempfile.NamedTemporaryFile(
            'w', suffix='.mp4', dir=os.path.dirname(video_fname)
        )
        tmp_dir = video_fname.replace('.mp4', '')
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_video_file.close()

        writer = cv2.VideoWriter(
            tmp_video_file.name,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (width, height), True,
        )

        # Compute per-video center from the first frame, apply Y offset
        centers = []
        for v in v_list:
            center = np.mean(v[0], axis=0)
            v[..., 1] = v[..., 1] - 0.125
            centers.append(center)

        r = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=800)

        # Load ASR data if provided
        asr_data = None
        if asr_path is not None:
            try:
                asr_data = np.load(asr_path)
            except Exception:
                asr_data = np.zeros((num_frames,))

        for i_frame in tqdm(range(num_frames), desc='Rendering frames'):
            cur_img = []
            for i in range(len(v_list)):
                if face:
                    img = render_mesh_helper(
                        Mesh(v_list[i][i_frame], self.template_mesh.f),
                        centers[i], r=r, xmag=0.15, y=0.95, z=1.0,
                        camera='o',
                    )
                else:
                    img = render_mesh_helper(
                        Mesh(v_list[i][i_frame], self.template_mesh.f),
                        centers[i], camera='o', r=r, y=0.7,
                    )
                cur_img.append(img)

            if num_video == 1:
                final_img = cur_img[0].astype(np.uint8)
            elif num_video == 2:
                final_img = np.hstack(
                    (cur_img[0], cur_img[1])
                ).astype(np.uint8)
            elif num_video == 3:
                final_img = np.hstack(
                    (cur_img[0], cur_img[1], cur_img[2])
                ).astype(np.uint8)
            elif num_video == 4:
                final_img = np.hstack(
                    (cur_img[0], cur_img[1], cur_img[2], cur_img[3])
                ).astype(np.uint8)

            if add_text:
                if asr_data is not None:
                    idx = min(i_frame, len(asr_data) - 1)
                    text = f'{asr_data[idx]} {i_frame}'
                else:
                    text = str(i_frame)
                final_img = add_image_text(
                    final_img, text, w=width, h=height
                ).astype(np.uint8)

            save_img_path = os.path.join(tmp_dir, '{:05d}.png'.format(i_frame))
            cv2.imwrite(save_img_path, final_img)

        writer.release()

        # Encode frames to video with ffmpeg
        image_sequence = os.path.join(tmp_dir, "%05d.png")
        cmd = [
            "ffmpeg", "-y",
            "-r", str(fps),
            "-i", image_sequence,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            tmp_video_file.name,
        ]
        call(cmd)

        # Merge audio if provided
        if cur_wav_file is not None:
            cmd = [
                'ffmpeg', '-y',
                '-i', tmp_audio_file.name,
                '-i', tmp_video_file.name,
                '-vcodec', 'h264',
                '-ac', '2',
                '-channel_layout', 'stereo',
                '-pix_fmt', 'yuv420p',
                video_fname,
            ]
            call(cmd)
            os.remove(tmp_audio_file.name)
        else:
            cmd = [
                'ffmpeg', '-y',
                '-i', tmp_video_file.name,
                '-vcodec', 'h264',
                '-pix_fmt', 'yuv420p',
                video_fname,
            ]
            call(cmd)

        os.remove(tmp_video_file.name)

        # Clean up temporary frame images
        shutil.rmtree(tmp_dir, ignore_errors=True)
