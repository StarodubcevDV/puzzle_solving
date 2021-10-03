import numpy as np
import os
import sys


W = 1200
H = 900
CHANNEL_NUM = 3
MAX_VALUE = 255
sides = ['r_side', 'l_side', 't_side', 'b_side']


def read_image(path):
    # second line of header contains image dimensions
    w, h = np.loadtxt(path, skiprows=1, max_rows=1, dtype=np.int32)
    # skip 3 lines reserved for header and read image
    image = np.loadtxt(path, skiprows=3, dtype=np.uint8).reshape((h, w, CHANNEL_NUM))
    return image


def write_image(path, img):
    h, w = img.shape[:2]
    # ppm format requires header in special format
    header = f'P3\n{w} {h}\n{MAX_VALUE}\n'
    with open(path, 'w') as f:
        f.write(header)
        for r, g, b in img.reshape((-1, CHANNEL_NUM)):
            f.write(f'{r} {g} {b} ')


def calc_distance(pix_pare: tuple):
    r_1 = float(pix_pare[0][0])
    g_1 = float(pix_pare[0][1])
    b_1 = float(pix_pare[0][2])
    r_2 = float(pix_pare[1][0])
    g_2 = float(pix_pare[1][1])
    b_2 = float(pix_pare[1][2])

    dist = (((r_2-r_1)**2)+((g_2-g_1)**2)+((b_2-b_1)**2)) ** 0.5
    return dist


def make_empty_img(tiles):
    shape = tiles.shape
    h_i, w_i = shape[1], shape[2]
    k_w = W / w_i
    k_h = H / h_i
    empty_img = np.zeros(shape=(k_h, k_w))
    return empty_img


def get_main_sides(img):
    res = {}
    r_main_side = np.array(img[0:-1, -1, :])
    l_comp_side = np.array(img[0:-1, 0, :])
    l_main_side = np.flip(l_comp_side, axis=0)
    t_main_side = np.array(img[0, 0:-1, :])
    b_comp_side = np.array(img[-1, 0:-1, :])
    b_main_side = np.flip(b_comp_side, axis=0)

    res[sides[0]] = r_main_side
    res[sides[1]] = l_main_side
    res[sides[2]] = t_main_side
    res[sides[3]] = b_main_side
    return res


def get_comp_sides(img):
    res = {}
    r_main_side = np.array(img[0:-1, -1, :])
    r_comp_side = np.flip(r_main_side, axis=0)
    l_comp_side = np.array(img[0:-1, 0, :])
    t_main_side = np.array(img[0, 0:-1, :])
    t_comp_side = np.flip(t_main_side, axis=0)
    b_comp_side = np.array(img[-1, 0:-1, :])

    res[sides[0]] = r_comp_side
    res[sides[1]] = l_comp_side
    res[sides[2]] = t_comp_side
    res[sides[3]] = b_comp_side
    return res


def check_sides(main_img, comp_img, comp_tile_id, res_dict):
    main_img_sides = get_main_sides(main_img)
    comp_img_sides = get_comp_sides(comp_img)
    for main_side in list(main_img_sides.keys()):
        for comp_side in list(comp_img_sides.keys()):
            dist = np.sum(list(map(calc_distance,
                                   zip(main_img_sides[main_side],
                                       comp_img_sides[comp_side])))) / main_img.shape[0]
            if dist < 50:
                res_dict[f'{main_side}'].append([comp_tile_id,
                                            comp_side,
                                           dist])
    return res_dict


def solve_puzzle(tiles_folder):
    # create placeholder for result image
    # read all tiles in list
    tiles = np.array([read_image(os.path.join(tiles_folder, t)) for t in sorted(os.listdir(tiles_folder))])
    print(tiles.shape)
    res_dict = {}
    for i in range(len(tiles)):
        res_dict[i] = {i: [] for i in sides}
        for j in range(len(tiles)):
            res_dict[i] = check_sides(tiles[i], tiles[j], j, res_dict[i])
    print(res_dict)

    result_img = np.zeros((H, W, CHANNEL_NUM), dtype=np.uint8)
    # scan dimensions of all tiles and find minimal height and width
    dims = np.array([t.shape[:2] for t in tiles])
    h, w = np.min(dims, axis=0)
    # compute grid that will cover image
    # spacing between grid rows = min h
    # spacing between grid columns = min w
    x_nodes = np.arange(0, W, w)
    y_nodes = np.arange(0, H, h)
    xx, yy = np.meshgrid(x_nodes, y_nodes)
    nodes = np.vstack((xx.flatten(), yy.flatten())).T
    # fill grid with tiles
    for (x, y), tile in zip(nodes, tiles):
        result_img[y: y + h, x: x + w] = tile[:h, :w]

    output_path = "image1.ppm"
    write_image(output_path, result_img)


if __name__ == "__main__":
    # directory = sys.argv[1]
    solve_puzzle('D:\\python_projects\\3divi_test\\data\\0000_0000_0000\\tiles')
