#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UI - a simple web search engine.
The goal is to index an infinite list of URLs (web pages),
and then be able to quickly search relevant URLs against a query.

See https://github.com/AnthonySigogne/web-search-engine for more information.
"""

__author__ = "Anthony Sigogne"
__copyright__ = "Copyright 2017, Byprog"
__email__ = "anthony@byprog.com"
__license__ = "MIT"
__version__ = "1.0"

import os
import requests
from urllib import parse
from flask import Flask, request, jsonify, render_template
import json
import sys
import random
import math

from bokeh.models import GraphRenderer, StaticLayoutProvider, Plot, Range1d, MultiLine, Circle, HoverTool, BoxZoomTool, ResetTool
from bokeh.palettes import Spectral4
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.embed import components
from bokeh.embed import json_item
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.sampledata.iris import flowers

sys.path.append('../search/')
from fuzzy_and_cluster import *
from segment import *

# init flask app and env variables
app = Flask(__name__)

searched = None

machine = pd.read_csv('./machine_now.csv')
machine = machine['0'].tolist()
issue = pd.read_csv('./issue_now.csv')
issue = issue['0'].tolist()
with open('./machine_dic.pickle', "rb") as fp: 
    ne_machine_dic = pickle.load(fp) 
with open('./issue_dic.pickle', "rb") as fp: 
    ne_issue_dic = pickle.load(fp) 

@app.route("/", methods=['GET'])
def search():
    # GET data
    query = request.args.get("query", None)
    filter_key = request.args.get("filter_key", '', None)
    start = request.args.get("start", 0, type=int)
    hits = request.args.get("hits", 10, type=int)
    filter_count = request.args.get("filter_count", 35, type=int)
    selected_filter = ''
    if start < 0 or hits < 0 :
        return "Error, start or hits cannot be negative numbers"

    global searched
    if query :
        r = None
        if filter_key =='':
        # query search engine
            try :
               searched = fuzzy_match_by_key(query)
               searched = filter_search(searched, selected_filter)
            except :
                return "Error, check your installation"
        else:
               searched = filter_search(searched, filter_key)

        data = searched 
        i = int(start/hits)
        maxi = 1+int(data["total"]/hits)
        range_pages = range(i-5,i+5 if i+5 < maxi else maxi) if i >= 6 else range(0,maxi if maxi < 10 else 10)
        print (hits, range_pages)

        # show the list of matching results
        #plot = make_plot('petal_width', 'petal_length')
        ne_machine, kb_machine, ne_issue, kb_issue = find_kb_by_sentence(query, machine, issue, ne_machine_dic, ne_issue_dic)
        print (ne_machine, kb_machine, ne_issue, kb_issue)
        #plot = render_kb('泵车', ['不走', '漏油', '漏电', '卡住', '显示故障', '莫名锁车'])
        plot_machine = render_kb('故障部位', ne_machine, kb_machine)
        script1, div1 = components(plot_machine)
        plot_issue = render_kb('故障描述', ne_issue, kb_issue)
        script2, div2 = components(plot_issue)

        return render_template('spatial/index.html', 
            query=query,
            script1=script1, 
            script2=script2, 
            div1=div1,
            div2=div2,
            #response_time=r.elapsed.total_seconds(),
            response_time=round(random.uniform(1,10)/10,2),
            total=data["total"],
            hits=hits,
            start=start,
            range_pages=range_pages,
            results=data["results"][i*hits:i*hits+hits],
            filters=data["filters"][:filter_count],
            page=i,
            maxpage=maxi-1)

    # return homepage (no query)
    return render_template('spatial/index.html')

from bokeh.models import PanTool,ZoomInTool,ZoomOutTool
def render_kb(name, entity, attribute):
    if attribute is None:
        N = 0
    else:
        N = len(attribute)    
    node_indices = list(range(N+1))
    plot = Plot(plot_width=2000, plot_height=2000,
                x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
    plot.add_tools(PanTool())
    plot.add_tools(ZoomInTool())
    plot.title.text = name

    graph = GraphRenderer()
    graph.node_renderer.data_source.add(node_indices, 'index')
    graph.node_renderer.glyph = Circle(size=50, fill_color=Spectral4[0])
    graph.edge_renderer.data_source.data = dict(
                start=[0]*N,
                end=node_indices[1:])

    circ = [i*2*math.pi/N for i in range(N)]
    r = [random.random() for i in circ]
    for i,x in enumerate(r):
        if x < 0.3:
            r[i]+= 0.3
    x = [0]+[math.cos(c)*r[i] for i,c in enumerate(circ)]
    y = [0]+[math.sin(c)*r[i] for i,c in enumerate(circ)]

    graph_layout = dict(zip(node_indices, zip(x, y)))
    graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

    attribute = [entity] +  attribute
    source = ColumnDataSource({'x': x, 'y': y,
        'txt': [ attribute[i] for i in range(len(attribute))]})
    labels = LabelSet(x='x', y='y', text='txt', x_offset=-20, y_offset=-15, source=source,text_font_size='18pt', background_fill_color='white')
    plot.renderers.append(labels)

    plot.renderers.append(graph)

    return plot


@app.route("/select", methods=['GET'])
def select():
    # GET data
    query = request.args.get("query", None)
    filter_key = request.args.get("filter_key", '', None)
    start = request.args.get("start", 0, type=int)
    hits = request.args.get("hits", 10, type=int)
    filter_count = request.args.get("filter_count", 35, type=int)
    selected_filter = ''
    if start < 0 or hits < 0 :
        return "Error, start or hits cannot be negative numbers"

    global searched

    # get data and compute range of results pages
    #data = r.json()
    #print (r, file=sys.stderr)
    data = searched
    i = int(start/hits)
    maxi = 1+int(data["total"]/hits)
    range_pages = range(i-5,i+5 if i+5 < maxi else maxi) if i >= 6 else range(0,maxi if maxi < 10 else 10)
    print (hits, range_pages)

    # show the list of matching results
    return render_template('spatial/index.html', query=query,
        #response_time=r.elapsed.total_seconds(),
        response_time=1.14,
        total=data["total"],
        hits=hits,
        start=start,
        range_pages=range_pages,
        results=data["results"][i*hits:i*hits+hits],
        filters=data["filters"][:filter_count],
        page=i,
        maxpage=maxi-1)

@app.route("/reference", methods=['POST'])
def reference():
    """
    URL : /reference
    Request the referencing of a website.
    Method : POST
    Form data :
        - url : url to website
        - email : contact email
    Return homepage.
    """
    # POST data
    data = dict((key, request.form.get(key)) for key in request.form.keys())
    if not data.get("url", False) or not data.get("email", False) :
        return "Vous n'avez pas renseigné l'URL ou votre email."

    # query search engine
    try :
        r = requests.post('http://%s:%s/reference'%(host, port), data = {
            'url':data["url"],
            'email':data["email"]
        })
    except :
        return "Une erreur s'est produite, veuillez réessayer ultérieurement"

    return "Votre demande a bien été prise en compte et sera traitée dans les meilleurs délais."

# -- JINJA CUSTOM FILTERS -- #

@app.template_filter('truncate_title')
def truncate_title(title):
    """
    Truncate title to fit in result format.
    """
    return title if len(title) <= 70 else title[:70]+"..."

@app.template_filter('truncate_description')
def truncate_description(description):
    """
    Truncate description to fit in result format.
    """
    if len(description) <= 160 :
        return description

    cut_desc = ""
    character_counter = 0
    for i, letter in enumerate(description) :
        character_counter += 1
        if character_counter > 160 :
            if letter == ' ' :
                return cut_desc+"..."
            else :
                return cut_desc.rsplit(' ',1)[0]+"..."
        cut_desc += description[i]
    return cut_desc

@app.template_filter('truncate_url')
def truncate_url(url):
    """
    Truncate url to fit in result format.
    url = parse.unquote(url)
    if len(url) <= 60 :
        return url
    url = url[:-1] if url.endswith("/") else url
    url = url.split("//",1)[1].split("/")
    url = "%s/.../%s"%(url[0],url[-1])
    return url[:60]+"..." if len(url) > 60 else url
    """
    return url

if __name__ == '__main__':
    load_data()
    app.run(host="0.0.0.0", debug = True, port=5000)
