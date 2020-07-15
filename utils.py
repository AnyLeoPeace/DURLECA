import numpy as np



'''Load true data'''
def load_data(return_locations = False, noise = 0):
    data = np.load('../data/processed_data.npy', allow_pickle=True).item()
    location_xys = data['xys']
    location_lat_lng = data['lat_lng']
    population = data['population'][:, :-1].transpose(1,0).astype(np.uint16)
    OD = data['od'] 
    locations_xy, locations_index = get_locations(location_xys)
    population_max = np.argmax(population.sum(axis=1))
    population = population[population_max] # 用于初始化人口
    # 
    if return_locations:
        return OD, population, locations_xy, locations_index
    else:
        return OD, population




def get_neighbor(OD_time, ox, oy, locations_xy, locations_index, neighbor = 1):
    '''
    return the OD flows to neighborhoods
    '''
    oidx = str([ox, oy])
    oidx = locations_index[oidx]
    xys = []
    ods = []
    for didx in range(323):
        dx, dy = locations_xy[didx]
        flag = False
        if OD_time[oidx, didx] > 0:
            flag  = True

        if neighbor:
            ddx = dx - ox
            ddy = dy - oy
            if (ddx not in range(-neighbor,neighbor+1)) or (ddy not in range(-neighbor,neighbor+1)):
                flag = False

        if flag:
            xys.append([dx,dy])
            ods.append(OD_time[oidx, didx])
    
    return xys, ods

def get_neighbor_OD(OD_time, locations_xy, locations_index, neighbor = 1):
    '''
    Input: 
        OD_time: 323 * 323, the OD matrix at one time step
        locations_xy: dict
        locations_index: dict
        neighbor: int, how many neighborhoods to calculate
    '''
    k = (2*neighbor+1)*(2*neighbor+1)
    flags = np.ones(k, bool)
    flags[k//2] = False

    OD_n = np.zeros((17, 19, k))
    print('Calculate',str(k-1)+'-neighborhood')

    for ox in range(17):
        for oy in range(19):

            oidx = locations_index[str([ox, oy])]
            xys = []
            ods = []
            
            for ddx in range(-neighbor,neighbor+1):
                for ddy in range(-neighbor,neighbor+1):

                    dx = ox + ddx
                    dy = oy + ddy

                    if str([dx, dy]) in locations_index:
                        didx = locations_index[str([dx, dy])]
                        OD_n[ox, oy, (ddx + neighbor)*(2*neighbor+1) + (ddy+neighbor)] = OD_time[oidx, didx]

    return OD_n[:,:,flags]



def get_locations(location_xys):
    locations_xy = {}

    for index, item in enumerate(location_xys):
        x, y = item.split('-')
        x = int(x)
        y = int(y)
        locations_xy[index] = [x,y]

    locations_index = {str(locations_xy[k]):k for k in locations_xy}

    return locations_xy, locations_index


def xys_to_xs(locations_xy):
    return [locations_xy[key][0]*17 + locations_xy[key][1] for key in locations_xy]


def get_connected(OD):
    N = OD.shape[1]
    connected_ori = [[]] * N
    connected_des = [[]] * N
    for i in range(len(OD)):
        connected_ori, connected_des = get_connected_once(connected_ori, connected_des, OD[i])
    
    return connected_ori, connected_des


def get_connected_once(connected_ori, connected_des, adj):
    '''
    Input current connected list and adjacent matrix
    Output next connected list
    '''
    N = len(connected_ori)

    for i in range(N):
        connected_ori[i] = connected_ori[i] + list(np.where(adj[i])[0])
        connected_des[i] = connected_des[i] + list(np.where(adj[:,i])[0])
    
    return connected_ori, connected_des


def simulation_round(x):
    sign = np.sign(x)
    x = abs(x)
    is_up = np.random.rand() < x- x.astype(np.int32)
    xx = np.where(is_up, np.ceil(x), np.floor(x))
    return xx.astype(np.int32)