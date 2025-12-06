import os
import pymeshlab as ml


def ply_to_obj_with_texture(
    input_ply_path,
    output_obj_path,
    texture_png_path=None,
    target_faces=300_000,
    texdim=4096,
):
    """
    从带 vertex color 的 .ply 生成:
      - 带 UV + 纹理的 .obj
      - 对应的 .mtl
      - 由 vertex color 烤出来的 texture.png
    """

    if texture_png_path is None:
        # 默认和 obj 放在同一路径下
        base, _ = os.path.splitext(output_obj_path)
        texture_png_path = base + "_texture.png"

    ms = ml.MeshSet()
    ms.load_new_mesh(input_ply_path)

    m = ms.current_mesh()
    print("Vertices:", m.vertex_number())
    print("Faces:", m.face_number())
    print("Has vertex colors:", m.has_vertex_color())

    if not m.has_vertex_color():
        raise RuntimeError("当前 PLY 没有 vertex color，没法从颜色生成 texture。")

    # ---- 1. 适当 decimate，避免 UV 展开时报 Inter-Triangle border is too much ----
    if m.face_number() > target_faces:
        print(f"[Decimate] faces {m.face_number()} -> {target_faces}")
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            preserveboundary=True,
            preservenormal=True,
            preservetopology=True,
            optimalplacement=True,
        )
        m = ms.current_mesh()
        print("Faces after decimate:", m.face_number())

    # ---- 2. UV 参数化（trivial triangle per wedge）----
    # 对应 MeshLab 里的: 
    # Parametrization: Trivial Per-Triangle (per-wedge)
    print("[UV] computing trivial per-wedge parametrization...")
    ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(
        textdim=texdim,   # 这个是 UV 展开的 target 尺寸，和 texture 大小逻辑相关
        # 其他参数用默认即可
    )

    # ---- 3. 从 vertex color 烤出纹理 ----
    # 注意：这里 **不要** 传 overwrite=True
    # 官方示例也是只传 textname：ms.compute_texmap_from_color(textname='xxx.png')
    print("[Texture] baking texture from vertex colors...")
    ms.compute_texmap_from_color(
        textname=os.path.basename(texture_png_path),
        # 让它把纹理图片写到当前工作目录；真正的路径由 save_current_mesh 控制
        # 如果你希望绝对路径，可以先 chdir 或者之后把文件移过去
    )

    # ---- 4. 保存 OBJ（会同时写 .mtl，并在 .mtl 中引用上面的 PNG）----
    # 一般不需要额外参数，默认会把 texcoord 和 texture 都写出去
    print("[Save] saving OBJ with MTL and texture...")
    ms.save_current_mesh(
        output_obj_path,
        save_vertex_color=False,   # 既然已经烤成纹理了，可以不用再保存 vertex color
        save_wedge_texcoord=True,  # 保存 per-wedge UV
    )

    print("Done.")
    print("OBJ:", output_obj_path)
    print("Texture (png):", texture_png_path)


if __name__ == "__main__":
    ply_to_obj_with_texture(
        "data/sam3d_mustard.ply",
        "data/sam3d_mustard_tex.obj",
    )
