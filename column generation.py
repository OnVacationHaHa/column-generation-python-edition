import os, csv, math
import numpy as np


class read:
    @staticmethod
    def __read_file(path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                lines[i] = lines[i].replace('\n', '')
                lines[i] = lines[i].split(',')
        f.close
        return lines

    @staticmethod
    def time_sub(t1, t2):
        h1 = t1 // 100
        h2 = t2 // 100
        m1 = t1 % 100
        m2 = t2 % 100
        return (h1 - h2) * 60 + m1 - m2

    @staticmethod
    def time_add2(t1, t2):
        h1 = t1 // 100
        m1 = t1 % 100
        if m1 + t2 < 60:
            return h1 * 100 + (m1 + t2)
        else:
            add_h = (m1 + t2) // 60
            add_m = (m1 + t2) % 60
            return (h1 + add_h) * 100 + add_m

    @staticmethod
    def __read_train():
        path = os.path.dirname(os.path.realpath(__file__)) + '\\input\\train running time.csv'
        lines = read.__read_file(path)
        del lines[0]
        train_running_time_list = []
        for l in lines:
            t = []
            for i in range(2, len(l) + 1):
                t.append(int(l[i - 1]))
            train_running_time_list.append(t)
        return (train_running_time_list, len(train_running_time_list[0]) + 1)

    @staticmethod
    def read_input_path():
        path = os.path.dirname(os.path.realpath(__file__)) + '\\input_path\\input_path.csv'
        lines = read.__read_file(path)
        del lines[0]
        input_path_list = []
        for l in lines:
            now_path = []
            now_path.append(int(l[1]))
            node_seq = l[2].split(';')
            time_seq = l[3].split(';')
            now_path.append(node_seq)
            now_path.append(time_seq)
            input_path_list.append(now_path)
        return input_path_list

    @staticmethod
    def __read_time():
        path = os.path.dirname(os.path.realpath(__file__)) + '\\input\\time information.csv'
        line = read.__read_file(path)
        del line[0]
        for i in range(len(line[0])):
            line[0][i] = int(line[0][i])
        return line[0]

    @staticmethod
    def __read_max_waiting_time():
        path = os.path.dirname(os.path.realpath(__file__)) + '\\input\\max waiting time.csv'
        lines = read.__read_file(path)
        del lines[0]
        max_waiting_time_list = []
        for l in lines:
            t = []
            for i in range(2, len(l) + 1):
                t.append(int(l[i - 1]))
            max_waiting_time_list.append(t)
        return max_waiting_time_list

    @staticmethod
    def __read_min_waiting_time():
        path = os.path.dirname(os.path.realpath(__file__)) + '\\input\\min waiting time.csv'
        lines = read.__read_file(path)
        del lines[0]
        min_waiting_time_list = []
        for l in lines:
            t = []
            for i in range(2, len(l) + 1):
                t.append(int(l[i - 1]))
            min_waiting_time_list.append(t)
        return min_waiting_time_list

    @staticmethod
    def __read_departure_time():
        path = os.path.dirname(os.path.realpath(__file__)) + '\\input\\departure time range.csv'
        lines = read.__read_file(path)
        del lines[0]
        departure_list = []
        for l in lines:
            t = []
            for i in range(1, 3):
                t.append(int(l[i]))
            departure_list.append(t)
        return departure_list

    @staticmethod
    def __read_blocking_time():
        path = os.path.dirname(os.path.realpath(__file__)) + '\\input\\blocking time.csv'
        lines = read.__read_file(path)
        del lines[0]
        blocking_time = []
        for l in lines:
            del l[0]
            now_train = []
            for b in l:
                now_train.append(int(b))
            blocking_time.append(now_train)
        return blocking_time

    @staticmethod
    def __read_sequence_of_stops():
        path = os.path.dirname(os.path.realpath(__file__)) + '\\input\\sequence of stops.csv'
        lines = read.__read_file(path)
        del lines[0]
        stop_sequence = []
        for l in lines:
            t = []
            if l[1] != '':
                l[1] = l[1].split(';')
                for i in l[1]:
                    t.append(int(i))
            else:
                t.append(0)
            stop_sequence.append(t)
        return stop_sequence

    @staticmethod
    def __read_zone():
        path = os.path.dirname(os.path.realpath(__file__)) + '\\input\\zone.csv'
        lines = read.__read_file(path)
        del lines[0]
        zone_list = []
        for l in lines:
            zone_s_time = int(l[1])
            zone_e_time = int(l[2])
            zone_station = int(l[3])
            zone_list.append([zone_s_time, zone_e_time, zone_station])
        return zone_list

    @staticmethod
    def __read_type():
        path = os.path.dirname(os.path.realpath(__file__)) + '\\input\\train type.csv'
        lines = read.__read_file(path)
        del lines[0]
        train_type = []
        for l in lines:
            l[1] = l[1].split(';')
            for i in l[1]:
                train_type.insert(int(i) - 1, l[0].replace('\t', ''))
        return train_type

    def __init__(self):
        (self.running_time, self.station_num) = read.__read_train()
        self.train_num = len(self.running_time)
        [self.t_s_time, self.t_p_time, self.s_time, self.e_time] = read.__read_time()
        self.time_len = read.time_sub(self.e_time, self.s_time) + 1
        self.sum_node = self.time_len * ((self.station_num - 2) * 3 + 2)
        self.departure_list = read.__read_departure_time()
        self.stop_list = read.__read_sequence_of_stops()
        self.max_waiting_time_list = read.__read_max_waiting_time()
        self.block_time = read.__read_blocking_time()
        self.min_waiting_time_list = read.__read_min_waiting_time()
        self.zone = read.__read_zone()
        self.train_type = read.__read_type()


class train:
    def __init__(self, r, train_id):
        self.r = r
        self.arr_tf = r.block_time[train_id - 1][0]
        self.arr_tp = r.block_time[train_id - 1][1]
        self.dep_tf = r.block_time[train_id - 1][2]
        self.dep_tp = r.block_time[train_id - 1][3]
        self.pass_tf = r.block_time[train_id - 1][4]
        self.pass_tp = r.block_time[train_id - 1][5]
        self.total_num = ((r.station_num - 2) * 3 + 2) * r.time_len + 2
        self.train_id = train_id
        self.is_init = True

    @staticmethod
    def __Dijkstra(D, s, t):
        i = s
        r = D.shape[0]
        List = list(range(1, r + 1))
        pred = [s] * r
        d = [float("inf")] * r
        d[i - 1] = 0
        pred[i - 1] = i
        del List[i - 1]
        while len(List) != 0:
            for k in range(1, len(List) + 1):
                j = List[k - 1]
                if d[j - 1] > d[i - 1] + D[i - 1, j - 1]:
                    d[j - 1] = d[i - 1] + D[i - 1, j - 1]
                    pred[j - 1] = i
            d_temp = []
            for l in List:
                d_temp.append(d[l - 1])
            index = d_temp.index(min(d_temp))
            i = List[index]
            del List[index]
        if d[t - 1] != float('inf'):
            Path = [t]
            now_node = t
            while now_node != s:
                pred_node = pred[now_node - 1]
                Path.insert(0, pred_node)
                now_node = pred_node
            del Path[0]
            del Path[len(Path) - 1]
        else:
            Path = float('inf')
        return Path

    def return_time(self, path_, station):
        path = self.path_node2path_sst(path_)
        for i in range(len(path)):
            if path[i][0] == station:
                if path[i][1] == 1 or path[i][1] == 2:
                    if station != 1 and station != self.r.station_num:
                        arr_time = path[i][2]
                        dep_time = path[i + 1][2]
                        return (arr_time, dep_time)
                    elif station == 1:
                        arr_time = 0
                        dep_time = path[i][2]
                        return (arr_time, dep_time)
                    else:
                        dep_time = 0
                        arr_time = path[i][2]
                        return (arr_time, dep_time)
                else:
                    arr_time = path[i][2]
                    dep_time = path[i][2]
                    return (arr_time, dep_time)

    def path_node2path_sst(self, path):
        path_sst = []
        for i in range(len(path)):
            path_sst.append(self.__node_num2sst(path[i]))
        return path_sst

    def __node_num2sst(self, node_num):
        num = (node_num - 2) // self.r.time_len
        time = (node_num - 2) % self.r.time_len
        if time == 0:
            time = self.r.time_len
        if num >= 1:
            station = (num - 1) // 3 + 2
            state = num - 1 - (station - 2) * 3 + 1
        else:
            station = 1
            state = 2
        if station == self.r.station_num:
            state = 1
        return (station, state, time)

    def __get_tp_tf(self, state):
        if state == 1:
            tf = self.r.block_time[self.train_id - 1][0]
            tp = self.r.block_time[self.train_id - 1][1]
        elif state == 2:
            tf = self.r.block_time[self.train_id - 1][2]
            tp = self.r.block_time[self.train_id - 1][3]
        else:
            tf = self.r.block_time[self.train_id - 1][4]
            tp = self.r.block_time[self.train_id - 1][5]
        return (tf, tp)

    def column(self, path):
        col = np.zeros(((self.r.station_num - 1) * 2, self.r.time_len))
        for i in range(1, len(path) + 1):
            sst = self.__node_num2sst(path[i - 1])
            (tf, tp) = self.__get_tp_tf(sst[1])
            start_time = sst[2] - tf if sst[2] - tf > 1 else 1
            end_time = sst[2] + tp if sst[2] + tp < self.r.time_len else self.r.time_len
            for t in range(start_time, end_time):
                if sst[0] == 1:
                    col[0, t - 1] = 1
                elif sst[0] == self.r.station_num:
                    col[(self.r.station_num - 1) * 2 - 1, t - 1] = 1
                else:
                    if sst[1] == 1:
                        col[(sst[0] - 2) * 2 + 2 - 1, t - 1] = 1
                    elif sst[1] == 2:
                        col[(sst[0] - 2) * 2 + 3 - 1, t - 1] = 1
                    else:
                        col[(sst[0] - 2) * 2 + 2 - 1, t - 1] = 1
                        col[(sst[0] - 2) * 2 + 3 - 1, t - 1] = 1
        return col

    def sst2node_num(self, station, state, time):
        if station == 1:
            return time + 2
        elif station == self.r.station_num:
            return (station - 2) * 3 * self.r.time_len + self.r.time_len + time + 2
        else:
            return (station - 2) * 3 * self.r.time_len + self.r.time_len + (state - 1) * self.r.time_len + time + 2

    def cost(self, path):
        sst2 = self.__node_num2sst(path[len(path) - 1])
        return sst2[2]

    def __init_network_with_miu(self, miu, start_time):
        network = np.zeros((self.total_num, self.total_num))
        for i in range(self.total_num):
            for j in range(self.total_num):
                if i == j:
                    continue
                network[i, j] = float('inf')
        for now_time in range(start_time, self.r.time_len + 1):
            now_node = self.sst2node_num(1, 2, now_time)
            network[0, now_node - 1] = now_time
            (tf, tp) = self.__get_tp_tf(2)
            for tt in range(now_time - tf, now_time + tp):
                if tt < 1:
                    network[0, now_node - 1] -= miu[0, 0]
                    continue
                if tt > self.r.time_len:
                    network[0, now_node - 1] -= miu[0, self.r.time_len - 1]
                    continue
                network[0, now_node - 1] -= miu[0, tt - 1]
        for now_time in range(1, self.r.time_len + 1):
            now_node = self.sst2node_num(self.r.station_num, 1, now_time)
            network[now_node - 1, 1] = 0
        for i in range(1, self.r.station_num):
            now_station = i
            next_station = i + 1
            running_time = self.r.running_time[self.train_id - 1][i - 1]
            now_stop = False
            next_stop = False
            if now_station in self.r.stop_list[self.train_id - 1]:
                now_stop = True
            if next_station in self.r.stop_list[self.train_id - 1]:
                next_stop = True
            self.link_station(network, now_station, next_station, running_time, now_stop, next_stop, miu)
        for i in range(2, self.r.station_num):
            now_station = i
            for now_time in range(1, self.r.time_len + 1):
                min_wait_time = self.r.min_waiting_time_list[self.train_id - 1][i - 2]
                if min_wait_time == 0:
                    min_wait_time = 1
                max_wait_time = self.r.max_waiting_time_list[self.train_id - 1][i - 2]
                now_node = self.sst2node_num(now_station, 1, now_time)
                end_time = now_time + max_wait_time if now_time + max_wait_time < self.r.time_len else self.r.time_len
                for next_time in range(now_time + min_wait_time, end_time + 1):
                    next_node = self.sst2node_num(now_station, 2, next_time)
                    network[now_node - 1, next_node - 1] = next_time - now_time
                    for tt in range(next_time - tf, next_time + tp):
                        ii = (now_station - 2) * 2 + 3
                        if tt < 1:
                            network[0, now_node - 1] -= miu[ii - 1, 0]
                            continue
                        if tt > self.r.time_len:
                            network[0, now_node - 1] -= miu[ii - 1, self.r.time_len - 1]
                            continue
                        network[now_node - 1, next_node - 1] -= miu[ii - 1, tt - 1]
        return network

    def link_station(self, network, now_station, next_station, running_time0, now_stop, next_stop, miu):
        now_states = []
        next_states = []
        if now_stop == False and now_station != 1:
            now_states.extend([2, 3])
        else:
            now_states.append(2)
        if next_stop == False and next_station != self.r.station_num:
            next_states.extend([1, 3])
        else:
            next_states.append(1)
        for now_time in range(1, self.r.time_len + 1):
            for now_state in now_states:
                running_time1 = running_time0
                now_node = self.sst2node_num(now_station, now_state, now_time)
                if now_state == 2:
                    running_time1 += self.r.t_s_time
                for next_state in next_states:
                    (tf, tp) = self.__get_tp_tf(next_state)
                    running_time = running_time1
                    if next_state == 1:
                        running_time += self.r.t_p_time
                    next_time = now_time + running_time
                    if next_time > self.r.time_len:
                        break
                    next_node = self.sst2node_num(next_station, next_state, next_time)
                    network[now_node - 1, next_node - 1] = running_time
                    for tt in range(next_time - tf, next_time + tp):
                        ii = (next_station - 2) * 2 + 2
                        if tt < 1:
                            network[0, now_node - 1] -= miu[ii - 1, 0]
                            continue
                        if tt > self.r.time_len:
                            network[0, now_node - 1] -= miu[ii - 1, self.r.time_len - 1]
                            continue
                        network[now_node - 1, next_node - 1] -= miu[ii - 1, tt - 1]

    def get_path(self, miu):
        if self.is_init == False:
            network = self.__init_network_with_miu(miu, 1)
        else:
            try:
                e_t = self.r.time_len // (self.r.train_num + 1)
                start_time = 1 + (self.train_id - 1) * e_t
                network = self.__init_network_with_miu(miu, start_time)
                self.is_init = False
            except Exception:
                print('Unsuccessfully solved,it may be that the input time range is not long enough')
        return train.__Dijkstra(network, 1, 2)

    def format_change(self, path):
        path_ = []
        now_station = 1
        dep_time = path[0][1]
        node = self.sst2node_num(now_station, 2, dep_time)
        now_station += 1
        path_.append(node)
        for i in range(1, len(path) - 1):
            arr_time = path[i][0]
            dep_time = path[i][1]
            if arr_time != dep_time:
                node1 = self.sst2node_num(now_station, 1, arr_time)
                node2 = self.sst2node_num(now_station, 2, dep_time)
                now_station += 1
                path_.append(node1)
                path_.append(node2)
            else:
                node = self.sst2node_num(now_station, 3, arr_time)
                now_station += 1
                path_.append(node)
        arr_time = path[len(path) - 1][0]
        node = self.sst2node_num(now_station, 1, arr_time)
        now_station += 1
        path_.append(node)
        return path_

    def generate_nexta(self, path_, train_id, node, road_link, agent, road_link_id,agent_id):
        path = self.format_change(path_)
        now_agent = [None, None, None, None, None, None, None, None, None, None, None, None, None, None]
        now_agent[0] = agent_id
        agent_id+=1
        now_agent[1] = train_id
        time_seq = ''
        node_seq = ''
        time_per = ''
        for i in range(len(path)):
            sst = self.__node_num2sst(path[i])
            node_row_num = (sst[0] - 1) * self.r.time_len + sst[2]
            zone_id = self.__get_zone_id(sst[0], sst[2])
            if i == 0:
                now_agent[2] = zone_id
                now_agent[4] = sst[0] * 1000000 + read.time_add2(self.r.s_time, sst[2] - 1)
                time_per += self.__time_int2string(read.time_add2(self.r.s_time, sst[2] - 1)) + "_"
                path_cost = sst[2]
            if i == len(path) - 1:
                now_agent[3] = zone_id
                now_agent[5] = sst[0] * 1000000 + read.time_add2(self.r.s_time, sst[2])
                time_per += self.__time_int2string(read.time_add2(self.r.s_time, sst[2] - 1))
                path_cost = sst[2] - path_cost
            time_seq += self.__time_int2string(read.time_add2(self.r.s_time, sst[2] - 1)) + ";"
            node_seq += str((sst[0] * 1000000 + read.time_add2(self.r.s_time, sst[2] - 1))) + ";"
            node[node_row_num][3] = zone_id
            node[node_row_num][4] = 1
        now_agent[6] = self.r.train_type[train_id - 1]
        now_agent[7] = time_per
        now_agent[8] = 1
        for i in range(9, 12):
            now_agent[i] = path_cost
        now_agent[12] = node_seq
        now_agent[13] = time_seq
        agent.append(now_agent)
        for i in range(len(path) - 1):
            now_node = self.__node_num2sst(path[i])
            next_node = self.__node_num2sst(path[i + 1])
            dr = [None, None, None, None, None, None, None, None, None, None, None, None]
            dr[1] = road_link_id
            road_link_id += 1
            from_node_id = (now_node[0] * 1000000) + read.time_add2(self.r.s_time, now_node[2] - 1)
            to_node_id = (next_node[0] * 1000000) + read.time_add2(self.r.s_time, next_node[2] - 1)
            cost = next_node[2] - now_node[2]
            dr[2] = from_node_id
            dr[3] = to_node_id
            dr[5] = 1
            dr[6] = cost
            for ii in range(7, 11):
                dr[ii] = 1
            dr[11] = cost
            road_link.append(dr)
        return (road_link_id,agent_id)

    def __time_int2string(self, t):
        if t < 1000:
            return "0" + str(t)
        else:
            return str(t)

    def __get_zone_id(self, now_station, now_time):
        if now_station != 1 and now_station != self.r.station_num:
            return 0
        now_time = read.time_add2(self.r.s_time, now_time)
        for i in range(len(self.r.zone)):
            check_station = r.zone[i][2]
            if now_time >= r.zone[i][0] and now_time <= r.zone[i][1] and now_station == check_station:
                return i + 1
        return 0


class CG:
    def __init__(self, r):
        self.r = r
        self.column_pool = [[] for i in range(self.r.train_num)]
        self.cost_pool = [[] for i in range(self.r.train_num)]
        self.path_pool = [[] for i in range(self.r.train_num)]
        self.resource_num = (r.station_num - 2) * 2 + 2
        self.miu = np.zeros((self.resource_num, r.time_len))
        self.sigma = np.zeros(r.train_num)

    def main(self):
        trains = []
        for i in range(self.r.train_num):
            trains.append(train(self.r, i + 1))
        self.__init_solution__(trains)
        any_neg = True
        iteration = 1
        while any_neg == True:
            print("*" * 20)
            print('The (%d)th iteration is underway' % iteration)
            iteration += 1
            dual_var = self.dual()
            num = 0
            for i in range(self.resource_num):
                for j in range(self.r.time_len):
                    self.miu[i, j] = dual_var[num]
                    num += 1
            self.sigma = np.zeros(self.r.train_num)
            for i in range(self.r.train_num):
                self.sigma[i] = dual_var[num]
                num += 1
            new_paths = []
            for i in range(self.r.train_num):
                new_paths.append(trains[i].get_path(self.miu))
            any_neg = False
            for i in range(self.r.train_num):
                col = trains[i].column(new_paths[i])
                cost = trains[i].cost(new_paths[i])
                reduced_cost = cost
                for j in range(self.resource_num):
                    for t in range(self.r.time_len):
                        reduced_cost -= self.miu[j, t] * col[j, t]
                reduced_cost -= self.sigma[i]
                if reduced_cost < 0:
                    self.column_pool[i].append(col)
                    self.cost_pool[i].append(cost)
                    self.path_pool[i].append(new_paths[i])
                    any_neg = True
        opt_solution = self.simplex()
        col_num = 0
        opt_solution_ = []
        for i in range(len(self.cost_pool)):
            ld = []
            opt_solution_.append(ld)
            for j in range(len(self.cost_pool[i])):
                opt_solution_[i].append(opt_solution[col_num])
                col_num += 1
        self.viable(opt_solution_, trains)
        self.__output_all_path_pool__(trains)

    def __init_solution__(self, trains):
        try:
            input_path = read.read_input_path()
            self.__input_path__(input_path, trains)
        except:
            print('*'*20)
            print('No input_path was found')
        for i in range(len(trains)):
            path = trains[i].get_path(self.miu)
            self.column_pool[i].append(trains[i].column(path))
            self.cost_pool[i].append(trains[i].cost(path))
            self.path_pool[i].append(path)

    def __input_path__(self, input_path, trains):
        for p in input_path:
            train_id = p[0]
            now_path = []
            for i in range(len(p[1])):
                [station, state] = p[1][i].split('_')
                station = int(station)
                state = int(state)
                time = p[2][i]
                time = int(time)
                time = time - self.r.s_time + 1
                node_num = trains[train_id - 1].sst2node_num(station, state, time)
                now_path.append(node_num)
            self.column_pool[train_id - 1].append(trains[train_id - 1].column(now_path))
            self.cost_pool[train_id - 1].append(trains[train_id - 1].cost(now_path))
            self.path_pool[train_id - 1].append(now_path)

    def dual(self):
        rows = self.resource_num * self.r.time_len
        cols = 0
        for cost in self.cost_pool:
            cols += len(cost)
        A = np.zeros((rows, cols))
        row = 0
        for i in range(self.resource_num):
            for t in range(self.r.time_len):
                col = 0
                for f in range(self.r.train_num):
                    for j in range(len(self.column_pool[f])):
                        A[row, col] = self.column_pool[f][j][i, t]
                        col += 1
                row += 1
        B = np.zeros(rows)
        for i in range(rows):
            B[i] = 1
        Aeq = np.zeros((self.r.train_num, cols))
        Beq = np.zeros(self.r.train_num)
        col = 0
        for i in range(self.r.train_num):
            for j in range(len(self.cost_pool[i])):
                Aeq[i, col] = 1
                col += 1
        for i in range(self.r.train_num):
            Beq[i] = 1
        F = np.zeros(cols)
        col = 0
        for i in range(len(self.cost_pool)):
            for j in range(len(self.cost_pool[i])):
                F[col] = self.cost_pool[i][j]
                col += 1
        from scipy import optimize
        A_ = np.vstack((A, Aeq))
        A = A_.T
        f_ = np.hstack((B, Beq))
        bounds = []
        for i in range(1, rows + self.r.train_num + 1):
            if i <= rows:
                bounds.append((-float('inf'), 0))
            else:
                bounds.append((-float('inf'), float('inf')))
        bounds = tuple(bounds)

        res = optimize.linprog(-f_, A_ub=A, b_ub=F, bounds=bounds)
        return res.x

    def simplex(self):
        rows = self.resource_num * self.r.time_len
        cols = 0
        for cost in self.cost_pool:
            cols += len(cost)
        A = np.zeros((rows, cols))
        row = 0
        for i in range(self.resource_num):
            for t in range(self.r.time_len):
                col = 0
                for f in range(self.r.train_num):
                    for j in range(len(self.column_pool[f])):
                        A[row, col] = self.column_pool[f][j][i, t]
                        col += 1
            row += 1
        B = np.zeros(rows)
        for i in range(rows):
            B[i] = 1
        Aeq = np.zeros((self.r.train_num, cols))
        Beq = np.zeros(self.r.train_num)
        col = 0
        for i in range(self.r.train_num):
            for j in range(len(self.cost_pool[i])):
                Aeq[i, col] = 1
                col += 1
        for i in range(self.r.train_num):
            Beq[i] = 1
        F = np.zeros(cols)
        bounds = []
        for i in range(cols):
            bounds.append((0, 1))
        bounds = tuple(bounds)
        from scipy import optimize
        res = optimize.linprog(F, A, B, Aeq, Beq, bounds)
        return res.x

    def viable(self, opt_solution, trains):
        solutions = []
        for i in range(self.r.train_num):
            solutions.append([])
            for j in range(self.r.station_num):
                now_station = j + 1
                arr_time0 = 0
                dep_time0 = 0
                for k in range(len(opt_solution[i])):
                    (arr_time, dep_time) = trains[i].return_time(self.path_pool[i][k], now_station)
                    arr_time0 += arr_time * opt_solution[i][k]
                    dep_time0 += dep_time * opt_solution[i][k]
                arr_time0 = math.floor(arr_time0)
                dep_time0 = math.floor(dep_time0)
                solutions[i].append([int(arr_time0), int(dep_time0)])

        for i in range(self.r.station_num):
            self.__sort_station__(solutions, i + 1)
        for i in range(self.r.station_num - 1):
            self.__sort_section__(solutions, i + 1)
        for i in range(self.r.station_num):
            self.__sort_station__(solutions, i + 1)
        self.__remove_interspace__(solutions)
        (node,road_link,agent,agent_type)=self.__init_csv__()
        road_link_id=1
        agent_id = 1
        for i in range(len(trains)):
            (road_link_id,agent_id) = trains[i].generate_nexta(solutions[i], i + 1, node, road_link, agent, road_link_id,agent_id)
        out_path = os.path.dirname(os.path.realpath(__file__)) + '\\output\\feasible result'
        is_exist=os.path.exists(out_path)
        if not is_exist:
            os.makedirs(out_path)
        self.__save_csv__(out_path,node,road_link,agent,agent_type)

    def __output_all_path_pool__(self,trains):
        path_pool=[]
        for i in range(self.r.train_num):
            now_train_pool=[]
            for p in self.path_pool[i]:
                one_path=[]
                for k in range(4):
                    (arr,dep)=trains[i].return_time(p,k+1)
                    one_path.append([arr,dep])
                now_train_pool.append(one_path)
            path_pool.append(now_train_pool)
        (node, road_link, agent, agent_type) = self.__init_csv__()
        road_link_id = 1
        agent_id = 1
        for t in range(len(path_pool)):
            for p in path_pool[t]:
                (road_link_id,agent_id) = trains[t].generate_nexta(p, t + 1, node, road_link, agent, road_link_id,agent_id)
        out_path = os.path.dirname(os.path.realpath(__file__)) + '\\output\\path pool'
        is_exist = os.path.exists(out_path)
        if not is_exist:
            os.makedirs(out_path)
        self.__save_csv__(out_path, node, road_link, agent, agent_type)




    def __save_csv__(self,out_path,node,road_link,agent,agent_type):
        with open(out_path + '\\node.csv', 'w+', newline='') as f:
            writer = csv.writer(f)
            for l in node:
                writer.writerow(l)
            f.close()
        with open(out_path + '\\road_link.csv', 'w+', newline='') as f:
            writer = csv.writer(f)
            for l in road_link:
                writer.writerow(l)
            f.close()
        with open(out_path + '\\agent.csv', 'w+', newline='') as f:
            writer = csv.writer(f)
            for l in agent:
                writer.writerow(l)
            f.close()
        with open(out_path + '\\agent_type.csv', 'w+', newline='') as f:
            writer = csv.writer(f)
            for l in agent_type:
                writer.writerow(l)
            f.close()

    def __init_csv__(self):
        node = []
        node.append(['name', 'physical_node_id', 'node_id', 'zone_id', 'node_type', 'control_type', 'x_coord',
                     'y_coord'])
        road_link = []
        road_link.append(['name', 'road_link_id', 'from_node_id', 'to_node_id', 'facility_type', 'dir_flag', 'length',
                          'lanes', 'capacity', 'free_speed', 'link_type', 'cost'])
        agent = []
        agent.append(
            ['agent_id', 'train_id', 'o_zone_id', 'd_zone_id', 'o_node_id', 'd_node_id', 'agent_type', 'time_period', 'volume', 'cost', 'travel_time', 'distance', 'node_sequence',
             'time_sequence'])
        agent_type = []
        agent_type.append(['agent_type', 'name'])
        all_type = []
        for i in self.r.train_type:
            if i in all_type:
                continue
            else:
                all_type.append(i)
        for t in all_type:
            agent_type.append([t, t])
        self.__init_node__(node)
        return (node,road_link,agent,agent_type)

    def __remove_interspace__(self, solutions):
        dep0 = []
        for i in range(len(solutions)):
            dep0.append(solutions[i][0][1])
        dep0.sort(reverse=False)
        if dep0[0] != 1:
            time_diff = dep0[0] - 1
        else:
            return
        for s in range(len(solutions)):
            for i in range(len(solutions[i])):
                for j in range(0, 2):
                    if solutions[s][i][j] != 0:
                        solutions[s][i][j] -= time_diff

    def __init_node__(self, node):
        for i in range(1, self.r.station_num + 1):
            for t in range(1, self.r.time_len + 1):
                dr = [None, None, None, None, None, None, None, None]
                dr[1] = i
                dr[2] = i * 1000000 + read.time_add2(self.r.s_time, t - 1)
                dr[3] = 0
                dr[4] = 0
                dr[6] = t * 100
                dr[7] = i * 1000
                node.append(dr)

    def __sort_station__(self, solutions, station):
        arr_solution = []
        dep_solution = []
        for i in range(self.r.train_num):
            arr_solution.append([i + 1, solutions[i][station - 1][0]])
            dep_solution.append([i + 1, solutions[i][station - 1][1]])
        if station != 1:
            arr_solution.sort(key=lambda x: x[1], reverse=False)
            for i in range(len(arr_solution) - 1):
                train_id1 = arr_solution[i][0]
                train_id2 = arr_solution[i + 1][0]
                arr_solution[i + 1][1] = self.__arr_interval__(train_id1, train_id2, arr_solution[i][1], arr_solution[i + 1][1])
            for i in range(len(arr_solution)):
                now_train = arr_solution[i][0]
                solutions[now_train - 1][station - 1][0] = arr_solution[i][1]
        if station != r.station_num:
            dep_solution.sort(key=lambda x: x[1], reverse=False)
            for i in range(len(dep_solution) - 1):
                train_id1 = dep_solution[i][0]
                train_id2 = dep_solution[i + 1][0]
                dep_solution[i + 1][1] = self.__dep_interval__(train_id1, train_id2, dep_solution[i][1], dep_solution[i + 1][1])
            if station != 1:
                for i in range(len(dep_solution)):
                    min_wait_time = self.r.min_waiting_time_list[i][station - 2]
                    if dep_solution[i][1] - arr_solution[i][1] < min_wait_time:
                        dep_solution[i][1] = arr_solution[i][1] + min_wait_time
            for i in range(len(dep_solution)):
                now_train = dep_solution[i][0]
                solutions[now_train - 1][station - 1][1] = dep_solution[i][1]

    def __sort_section__(self, solutions, start_station):
        dep_solution = []
        for i in range(self.r.train_num):
            dep_solution.append([i + 1, solutions[i][start_station - 1][1]])
        dep_solution.sort(key=lambda x: x[1], reverse=False)
        for i in range(self.r.train_num):
            now_train = dep_solution[i][0]
            running_time = self.r.running_time[now_train - 1][start_station - 1]
            if solutions[now_train - 1][start_station - 1][0] != solutions[now_train - 1][start_station - 1][1]:
                running_time += self.r.t_s_time
            if solutions[now_train - 1][start_station][0] != solutions[now_train - 1][start_station][1]:
                running_time += self.r.t_p_time
            if solutions[now_train - 1][start_station][0] - solutions[now_train - 1][start_station - 1][1] < running_time:
                time_diff = running_time - (solutions[now_train - 1][start_station][0] - solutions[now_train - 1][start_station - 1][1])
                solutions[now_train - 1][start_station][0] += time_diff
                solutions[now_train - 1][start_station][1] += time_diff

    def __arr_interval__(self, train_id1, train_id2, arr1, arr2):
        tf1 = self.r.block_time[train_id1 - 1][0]
        tp1 = self.r.block_time[train_id1 - 1][1]
        tf2 = self.r.block_time[train_id2 - 1][0]
        tp2 = self.r.block_time[train_id2 - 1][1]
        if arr1 + tp1 + tf2 > arr2:
            return arr1 + tp1 + tf2
        else:
            return arr2

    def __dep_interval__(self, train_id1, train_id2, dep1, dep2):
        tf1 = self.r.block_time[train_id1 - 1][0]
        tp1 = self.r.block_time[train_id1 - 1][1]
        tf2 = self.r.block_time[train_id2 - 1][0]
        tp2 = self.r.block_time[train_id2 - 1][1]
        if dep1 + tp1 + tf2 > dep2:
            return dep1 + tp1 + tf2
        else:
            return dep2


if __name__ == '__main__':
    r = read()
    try:
        r = read()
    except Exception:
        print('Please close the input file！！！！')
    c = CG(r)
    print("*" * 20)
    print('Starting to calculate')
    print("*" * 20)
    print('Please ignore the warnings issued by the linear programming solver')
    c.main()
    print("*" * 20)
    print('Successfully solved, please open NEXTA.exe to view the results')
