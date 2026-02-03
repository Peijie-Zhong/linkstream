from lago.lago import LinkStream, lago_communities
import csv
import pandas as pd
from lago.l_modularity_function import longitudinal_modularity
import time

def lago_community(link_stream_file,
                   nb_iter, 
                   is_stream_graph=False, 
                   info_logging=False,
                   record_time=False):
    '''
    :return: dynamic communities found by LAGO

    :param link_stream_file: the path of link stream file
    :param nb_iter: number of iterations
    :param info_logging: print info logging if True
    :param record_time: record using time if True
    '''
    df = pd.read_csv(link_stream_file, header=None, index_col=False,names=["source","destination","timestamp"], skiprows=1)
    time_links = df.values.tolist()
    my_linkstream = LinkStream(is_stream_graph=is_stream_graph)
    my_linkstream.add_links(time_links)

    start_time = time.perf_counter()
    dynamic_communities = lago_communities(
        my_linkstream,
        nb_iter=nb_iter, 
        )
    end_time = time.perf_counter()


    if info_logging:
        print(f"The link stream consists of {my_linkstream.nb_edges} temporal edges\
               (or time links) accross {my_linkstream.nb_nodes} nodes and \
                {my_linkstream.network_duration} time steps, \
                of which only {my_linkstream.nb_timesteps} contain activity.")
        print(f"{len(dynamic_communities)} dynamic communities have been found")
        long_mod_score = longitudinal_modularity(
            my_linkstream, 
            dynamic_communities,
            lex_type="MM"
            )

        print(f"Longitudinal Modularity score of {long_mod_score} ")
    if record_time:
        return dynamic_communities, end_time - start_time
    else:
        return dynamic_communities