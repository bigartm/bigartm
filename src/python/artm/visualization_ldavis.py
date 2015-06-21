import json
import jinja2
import os
import binascii

class TopicModelVisualization(object):
    def __init__(self, data):
        self.data = data

    def _generate_json(self):
        return json.dumps(self.data)
        
    def _repr_html_(self):
        random_figid = binascii.hexlify(os.urandom(16))
        html = TEMPLATE_NOTEBOOK.render(
            figid=random_figid,
            figure_json=self._generate_json(),
            d3_url=URL_D3,
            ldavis_url='ldavis.js',
            extra_css=LDAVIS_CSS,
        )
        return html
        
    def to_file(self, filename, title=None):
        if title is None:
            title = 'LDAvis Topic Model Visualization'
            
        with open('../artm/_js/ldavis.js') as f:
            js_code = f.read()
            
        html = TEMPLATE_PAGE.render(
            title=title,
            d3_url=URL_D3,
            ldavis_url='ldavis.js',
            data_json=self._generate_json(),
            extra_css=LDAVIS_CSS,
            js_code=js_code,
        )
        with open(filename, 'wt') as fout:
            fout.write(html)


URL_D3 = 'https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.js'

LDAVIS_CSS = """
path {
  fill: none;
  stroke: none;
}
.xaxis .tick.major {
    fill: black;
    stroke: black;
    stroke-width: 0.1;
    opacity: 0.7;
}
.slideraxis {
    fill: black;
    stroke: black;
    stroke-width: 0.4;
    opacity: 1;
}
text {
    font-family: sans-serif;
    font-size: 11px;
}
"""

TEMPLATE_NOTEBOOK = jinja2.Template("""
<div id="ldavis_{{figid}}"></div>
<style>{{ extra_css }}</style>
<script>
function ldavis_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}
figure_data = {{ figure_json }};
if(typeof(LDAvis) !== "undefined"){
   // already loaded: just create the figure
   var vis = new LDAvis("#ldavis_{{figid}}", figure_data);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/mpld3
   require.config({paths: {d3: "{{ d3_url[:-3] }}"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      ldavis_load_lib("{{ ldavis_url }}", function(){
         var vis = new LDAvis("#ldavis_{{figid}}", figure_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & mpld3
    ldavis_load_lib("{{ d3_url }}", function(){
         ldavis_load_lib("{{ ldavis_url }}", function(){
                 var vis = new LDAvis("#ldavis_{{figid}}", figure_data);
            })
         });
}
</script>
""")


TEMPLATE_PAGE = jinja2.Template("""
<!DOCTYPE html>
<html>
  <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <title>{{ title }}</title>
    <script src="{{ d3_url }}"></script>
    <script>{{ js_code }}</script>
    <style>
    {{ extra_css }}
    </style>
  </head>
  <body>
    <div id = "lda"></div>
    <script>
      data = {{ data_json }};
      var vis = new LDAvis("#lda", data);
    </script>
  </body>
</html>
""")