import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
import matplotlib
import random
import math
import numpy

# import community as community

def create_conn_random_graph(nodes,p):
    while  True:
        # G=nx.connected_watts_strogatz_graph(25, 2, 0.8, tries=100)
        G=nx.erdos_renyi_graph(nodes,p)
        if nx.is_connected(G):
            break
    G.remove_nodes_from(nx.isolates(G))
    sstt="Erdos-Renyi Random Graph with %i nodes and probability %.02f" %(nodes,p)
    return G, sstt


def draw_network(G,sstt,pos={},with_edgewidth=False,withLabels=True,pernode_dict={},labfs=10,valpha=0.4,ealpha=0.4,labelfont=20):


# GI = graph_dic[ract_dic[cnum[3]]]
# print "The number of actors in Macbeth's Act IV is", len(GI.nodes())
# print "The number of conversational relationships in Macbeth's Act IV is", len(GI.edges())

    G.remove_nodes_from(nx.isolates(G))
    # if with_weights:
    #     weights={(i[0],i[1]):i[2]['weight'] for i in G.edges(data=True) }#if all((i[0],i[1])) in G.nodes() }
    plt.figure(figsize=(12,12))
    # try:
    #     f=open('positions_of_Mc_Shake.dmp')
    #     pos_dict=pickle.load(f)
    #     pos =pos_dict[3]
    # except:
        
    #     pos=nx.spring_layout(G,scale=50)
    #     pos_dict[3]=pos
    if len(pos)==0:
        pos=nx.spring_layout(G,scale=50)


# pos=nx.spring_layout(G,scale=50)
# pos_dict[3]=pos
    # if:
    #     labels={i:v for v,i in pernode_dict.items() if i in G.nodes()}
    # else:
    #     labels={i:v for v,i in pernode_dict.items() if i in G.nodes()}
    if with_edgewidth:
        edgewidth=[]
        for (u,v,d) in G.edges(data=True):
            edgewidth.append(d['weight'])
    else:
        edgewidth=[1 for i in G.edges()]
    nx.draw_networkx_nodes(G,pos=pos,with_labels=False,alpha=0.3)
    if withLabels:
        if len(pernode_dict)>0:
            labels={i:v for v,i in pernode_dict.items() if i in G.nodes()}
            labe=nx.draw_networkx_labels(G,pos=pos,labels=labels,font_size=labelfont)
        else:
            labe=nx.draw_networkx_labels(G,pos=pos,font_size=labfs)
    nx.draw_networkx_edges(G,pos=pos,edge_color='b',width=edgewidth, alpha=0.2)#,edge_labels=weights,label_pos=0.2)


    # pos=nx.spring_layout(G,scale=50)
    # plt.figure(figsize=(12,12))
    # nx.draw_networkx_nodes(G,pos=pos,with_labels=withLabels,alpha=valpha)
    # if withLabels:
    #     labe=nx.draw_networkx_labels(G,pos=pos,font_size=labfs)
    # # nx.draw_networkx_edges(G,pos=pos,edge_color='b',alpha=ealpha)
    plt.title(sstt,fontsize=20)
    kk=plt.axis('off')
    return pos


def draw_centralities(G,centr,pos,with_edgewidth=False,withLabels=True,pernode_dict={},title_st='', labfs=10,valpha=0.4,ealpha=0.4):

    plt.figure(figsize=(12,12))
    if centr=='degree_centrality':
        cent=nx.degree_centrality(G)
        sstt='Degree Centralities'
        ssttt='degree centrality'
    elif centr=='closeness_centrality':
        cent=nx.closeness_centrality(G)
        sstt='Closeness Centralities'
        ssttt='closeness centrality'
    elif centr=='betweenness_centrality':
        cent=nx.betweenness_centrality(G)
        sstt='Betweenness Centralities'
        ssttt='betweenness centrality'
    elif centr=='eigenvector_centrality':
        cent=nx.eigenvector_centrality(G,max_iter=2000)
        sstt='Eigenvector Centralities'
        ssttt='eigenvector centrality'
    elif centr=='katz_centrality':
        phi = (1+math.sqrt(5))/2.0 # largest eigenvalue of adj matrix
        cent=nx.katz_centrality_numpy(G,1/phi-0.01)
        sstt='Katz Centralities'
        ssttt='Katz centrality'
    elif centr=='page_rank':
        cent=nx.pagerank(G)
        sstt='PageRank'
        ssttt='pagerank'
    cs={}
    nods_dici={v:k for k,v in pernode_dict.items()}
    for k,v in cent.items():
        if v not in cs:
            cs[v]=[k]
        else:
            cs[v].append(k)
    for k in sorted(cs,reverse=True):
        for v in cs[k]:
            print 'Node %s has %s = %.4f' %(nods_dici[v],ssttt,k)

    if withLabels:
        if len(pernode_dict)>1:
            labels={i:v for v,i in pernode_dict.items() if i in G.nodes()}
            labe=nx.draw_networkx_labels(G,pos=pos,labels=labels,font_size=20)
        else:
            labe=nx.draw_networkx_labels(G,pos=pos,font_size=labfs)
    nx.draw_networkx_nodes(G,pos=pos,nodelist=cent.keys(), #with_labels=withLabels,
                           node_size = [d*4000 for d in cent.values()],node_color=cent.values(),
                           cmap=plt.cm.Reds,alpha=valpha)
    if with_edgewidth:
        edgewidth=[]
        for (u,v,d) in G.edges(data=True):
            edgewidth.append(d['weight'])
    else:
        edgewidth=[1 for i in G.edges()]
    nx.draw_networkx_edges(G,pos=pos,edge_color='b',width=edgewidth, alpha=ealpha)
    plt.title(title_st+' '+ sstt,fontsize=20)
    kk=plt.axis('off')



def draw_centralities_subplots(G,pos,withLabels=True,labfs=10,valpha=0.4,ealpha=0.4,figsi=(12,12)):
    centList=['degree_centrality','closeness_centrality','betweenness_centrality',
    'eigenvector_centrality','katz_centrality','page_rank']
    cenLen=len(centList)
    plt.figure(figsize=figsi)
    for uu,centr in enumerate(centList):
        if centr=='degree_centrality':
            cent=nx.degree_centrality(G)
            sstt='Degree Centralities'
            ssttt='degree centrality'
        elif centr=='closeness_centrality':
            cent=nx.closeness_centrality(G)
            sstt='Closeness Centralities'
            ssttt='closeness centrality'
        elif centr=='betweenness_centrality':
            cent=nx.betweenness_centrality(G)
            sstt='Betweenness Centralities'
            ssttt='betweenness centrality'
        elif centr=='eigenvector_centrality':
            try:
                cent=nx.eigenvector_centrality(G,max_iter=2000)
                sstt='Eigenvector Centralities'
                ssttt='eigenvector centrality'
            except:
                continue

            
        elif centr=='katz_centrality':
            phi = (1+math.sqrt(5))/2.0 # largest eigenvalue of adj matrix
            cent=nx.katz_centrality_numpy(G,1/phi-0.01)
            sstt='Katz Centralities'
            ssttt='Katz centrality'
        elif centr=='page_rank':
            try:
                cent=nx.pagerank(G)
                sstt='PageRank'
                ssttt='pagerank'
            except:
                continue
        cs={}
        for k,v in cent.items():
            if v not in cs:
                cs[v]=[k]
            else:
                cs[v].append(k)
        nodrank=[]
        uui=0
        for k in sorted(cs,reverse=True):
            for v in cs[k]:

                if uui<5:
                    nodrank.append(v)
                    uui+=1

        #         print 'Node %s has %s = %.4f' %(v,ssttt,k)
        nodeclo=[]
        for k,v in cent.items():
            if k in  nodrank :
                nodeclo.append(v)
            else:
                nodeclo.append(0.)
        plt.subplot(1+cenLen/2.,2,uu+1).set_title(sstt)
        if withLabels:
            labe=nx.draw_networkx_labels(G,pos=pos,font_size=labfs)
        
        # print uu,sstt
        nx.draw_networkx_nodes(G,pos=pos,nodelist=cent.keys(), #with_labels=withLabels,
                               # node_size = [d*4000 for d in cent.values()],
                               node_color=nodeclo,
                               cmap=plt.cm.Reds,alpha=valpha)
        
        nx.draw_networkx_edges(G,pos=pos,edge_color='b', alpha=ealpha)
        plt.title(sstt,fontsize=20)
        kk=plt.axis('off')

def draw_assor_attr_subplots(G,pos,sstt,attr_dict_graph,lis_pret,label_font=10,titlefont=20):

        # import matplotlib
    # select_attribute=select_attributes.result
    print sstt
    plt.figure(figsize=(12,12))
    for uu,attr_ao in enumerate(attr_dict_graph):
        
    # print 'ASSORTATIVITY COEFFICIENT (HOMOPHILY)'

        att_asso_coef=nx.attribute_assortativity_coefficient(G,attr_dict_graph[attr_ao])
    # print lis_pret[attr_ao]+' = ', '%.4f' %att_asso_coef
    # print
    # print 'Mixing Matrix (unnormalized)'

        # list_of_attributes_dc=[]
        # for ndk in G.nodes(data=True):
        #     print ndk
        #     attr_ni=ndk[1][attr_dict_graph[attr_ao]]
        #     if attr_ni not in list_of_attributes_dc:
        #         print attr_ni
        #         list_of_attributes_dc.append(attr_ni)

    # print pd.DataFrame(nx.attribute_mixing_matrix(G,attr_dict_graph[attr_ao],normalized=False),
    #                    index=list_of_attributes_dc,
    #                   columns=list_of_attributes_dc)
        color_parti={}
        labels_n={}
        for nd in G.nodes(data=True):
            # print nd
            labels_n[nd[0]]=nd[1]['label']
        #     print nd
            if nd[1][attr_dict_graph[attr_ao]] not in color_parti:
                color_parti[nd[1][attr_dict_graph[attr_ao]]]=[nd[0]]
            else:
                color_parti[nd[1][attr_dict_graph[attr_ao]]].append(nd[0])
        color=color_parti.values()#[[0, 7, 12, 15, 17, 19], [1, 9, 14], [2, 6, 13], [8, 11, 18], [5, 10, 16],[4,3]]
        color_part={v:i for i,k in enumerate(color) for v in k}
        colorsl=[name for name,hex in matplotlib.colors.cnames.iteritems()]
        rcolo_part={}
        for i,k in color_part.items():
            if k not in rcolo_part:
                rcolo_part[k]=[i]
            else:
                rcolo_part[k].append(i)
        for nd in G.nodes():
            G.add_node(nd,attr_dict=G.node[nd],color=color_part[nd])
        col ={j:colorsl[i] for i,v in enumerate(color) for j in v}
        colors = [col[j] for j in G.nodes()]
        ed_col={i:[] for i in color_part.values()}
        ned_col=[]
        for edg in G.edges():
            ed=edg[0]
            de=edg[1]
            
            if color_part[ed]==color_part[de]:
                ed_col[color_part[ed]].append(edg)
            else:
                ned_col.append(edg)

        atas=nx.attribute_assortativity_coefficient(G,attr_dict_graph[attr_ao])

        sstta="\n%s assortativity = %.04f" %(lis_pret[attr_ao],atas)
        plt.subplot(1+len(lis_pret)/2.,2,uu+1).set_title(sstta)
        # pospos=plt.figure(figsize=(12,12))
        nx.draw_networkx_nodes(G,pos=pos,with_labels=False,node_color=colors, alpha=0.4)#node_color=colors
        labe=nx.draw_networkx_labels(G,pos=pos,labels=labels_n,font_size=label_font)
        nx.draw_networkx_edges(G,pos=pos,edgelist=ned_col, edge_color='b', alpha=0.5)
        for ke,va in ed_col.items():
            ed_colors=[colorsl[ke] for i in va]
            nx.draw_networkx_edges(G,pos,edgelist=va,width=8,alpha=0.5,edge_color=ed_colors)
        # pospos=plt.title(sstta,fontsize=titlefont)
        pospos=plt.axis('off')