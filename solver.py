import numpy as np
import os
import sys


W = 1200
H = 900
CHANNEL_NUM = 3
MAX_VALUE = 255
sides = ['r_side', 't_side', 'l_side', 'b_side']


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
    k_w = int(W / w_i)
    k_h = int(H / h_i)
    empty_img = np.zeros(shape=(k_h, k_w))
    return empty_img


def make_solve_matrix(tiles):
    shape = tiles.shape
    h_i, w_i = shape[1], shape[2]
    k_w = int(W / w_i)
    k_h = int(H / h_i)
    matrix = np.zeros(shape=(k_h, k_w, 2))
    k = 0
    for i in range(k_h):
        for j in range(k_w):
            matrix[i][j] = [k, 0]
            k += 1
    return matrix


def get_main_sides(img):
    res = {}
    r_main_side = np.array(img[0:-1, -1, :])
    l_comp_side = np.array(img[0:-1, 0, :])
    l_main_side = np.flip(l_comp_side, axis=0)
    t_main_side = np.array(img[0, 0:-1, :])
    b_comp_side = np.array(img[-1, 0:-1, :])
    b_main_side = np.flip(b_comp_side, axis=0)

    res[sides[0]] = r_main_side
    res[sides[1]] = t_main_side
    res[sides[2]] = l_main_side
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
    res[sides[1]] = t_comp_side
    res[sides[2]] = l_comp_side
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
            if dist < 40:
                res_dict[f'{main_side}'].append([comp_tile_id,
                                                 comp_side,
                                                 dist])
    return res_dict


def find_min_neighbour(num_tile, side, res_dict):
    if len(res_dict[num_tile][side]) != 0:
        min_side = res_dict[num_tile][side][0]
        for side in res_dict[num_tile][side]:
            if min_side[2] > side[2]:
                min_side = side
        return min_side
    else:
        return None


def get_num_rotates(main_side, comp_side):
    ind_main_side = sides.index(main_side)
    ind_comp_side = sides.index(comp_side)

    if (ind_main_side + 2) >= len(sides)-1:
        ind_find_side = ind_main_side - 2
    else:
        ind_find_side = ind_main_side + 2

    if ind_comp_side > ind_find_side:
        num_rotates = 4 - (ind_comp_side - ind_find_side)
    else:
        num_rotates = (ind_find_side - ind_comp_side)

    return num_rotates


def swap_elements(solve_matrix, ind_1, ind_2):
    tmp = np.copy(solve_matrix[ind_2[0]][ind_2[1]])
    solve_matrix[ind_2[0]][ind_2[1]] = np.copy(solve_matrix[ind_1[0]][ind_1[1]])
    solve_matrix[ind_1[0]][ind_1[1]] = np.copy(tmp)
    return solve_matrix


def shift_right(arr):
    arr = np.roll(arr, 1, axis=1)
    return arr


def shift_left(arr):
    arr = np.roll(arr, -1, axis=1)
    return arr


def shift_up(arr):
    arr = np.roll(arr, -1, axis=0)
    return arr


def shift_down(arr):
    arr = np.roll(arr, 1, axis=0)
    return arr


def find_index_by_elemnet(arr, el):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i][j][0] == el:
                return [i, j]


# def concat_tiles(tiles):
    # solve_matrix = make_solve_matrix(tiles)
    # res_dict = {}
    # for i in range(len(tiles)):
    #     res_dict[i] = {i: [] for i in sides}
    #     for j in range(len(tiles)):
    #         res_dict[i] = check_sides(tiles[i], tiles[j], j, res_dict[i])
    #
    # tile_ind = 0
    # side_ind = 0
    # taken_ind = []
    # while True:
    #     if tile_ind < len(tiles)-1:
    #         if tile_ind in taken_ind:
    #             tile_ind += 1
    #             continue
    #         else:
    #             min_tile = find_min_neighbour(tile_ind, side_ind, res_dict)
    #             ind_min_tile = min_tile[0]
    #             side_min_tile = min_tile[1]
    #             if find_min_neighbour(ind_min_tile, side_min_tile, res_dict)[0] != tile_ind:
    #                 side_ind += 1
    #                 continue
    #             ind_1 = np.where(solve_matrix == tile_ind)
    #             ind_1[1] += 1
    #             ind_2 = np.where(solve_matrix == min_tile)
    #             solve_matrix = np.copy(swap_elements(solve_matrix, ind_1, ind_2))
    #     else:
    #         break


def concat_tiles(tiles):
    solve_matrix = make_solve_matrix(tiles)
    res_dict = {}
    for i in range(len(tiles)):
        res_dict[i] = {i: [] for i in sides}
        for j in range(len(tiles)):
            res_dict[i] = check_sides(tiles[i], tiles[j], j, res_dict[i])
    ind_side_full = {n: [] for n in list(res_dict.keys())}
    for i in list(res_dict.keys()):
        for j in sides:
            if j not in ind_side_full[i]:
                min_tile = find_min_neighbour(i, j, res_dict)
                if min_tile is None:
                    continue
                min_min_tile = find_min_neighbour(min_tile[0], min_tile[1], res_dict)
                if min_min_tile is None and min_min_tile[0] != i:
                    continue
                if min_tile[1] not in ind_side_full[min_tile[0]]:
                    rotates = get_num_rotates(j, min_tile[1])
                    ind_1 = find_index_by_elemnet(solve_matrix, i)
                    if j == 'r_side':
                        if ind_1[1] + 1 > solve_matrix.shape[1] - 1:
                            solve_matrix = np.copy(shift_left(solve_matrix))
                        else:
                            ind_1[1] += 1
                    elif j == 'l_side':
                        if ind_1[1] - 1 < 0:
                            solve_matrix = np.copy(shift_right(solve_matrix))
                        else:
                            ind_1[1] -= 1
                    elif j == 't_side':
                        if ind_1[0] - 1 < 0:
                            solve_matrix = np.copy(shift_down(solve_matrix))
                        else:
                            ind_1[0] -= 1
                    elif j == 'b_side':
                        if ind_1[0] + 1 > solve_matrix.shape[0] - 1:
                            solve_matrix = np.copy(shift_up(solve_matrix))
                        else:
                            ind_1[0] += 1
                    ind_2 = find_index_by_elemnet(solve_matrix, min_tile[0])
                    solve_matrix = np.copy(swap_elements(solve_matrix, ind_1, ind_2))
                    solve_matrix[ind_1[0]][ind_1[1]][1] += rotates
                    ind_side_full[i].append(j)
                    ind_side_full[min_tile[0]].append(min_tile[1])
    return solve_matrix


def solve_puzzle(tiles_folder):
    # create placeholder for result image
    # read all tiles in list
    tiles = np.array([read_image(os.path.join(tiles_folder, t)) for t in sorted(os.listdir(tiles_folder))])
    solve_matrix = concat_tiles(tiles).reshape(1, len(tiles), 2)
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
    index = 0
    for (x, y), tile in zip(nodes, tiles):
        i = int(solve_matrix[0][index][0])
        tile = np.rot90(tiles[i], int(solve_matrix[0][index][1]))
        result_img[y: y + h, x: x + w] = tile[:h, :w]
        index += 1
    output_path = "image.ppm"
    write_image(output_path, result_img)


if __name__ == "__main__":
    directory = sys.argv[1]
    solve_puzzle(directory)
