"use strict";

function addScript(src) {
    const script = document.createElement("script");
    // const materializeJS = document.createElement("script");
    const materialize = document.createElement("link");
    const icons = document.createElement("link");
    materialize.rel="stylesheet";
    materialize.href = chrome.extension.getURL("src/css/materialize.min.css");
    icons.href = "https://fonts.googleapis.com/icon?family=Material+Icons";
    icons.rel = "stylesheet";
    // materializeJS.type = "text/javascript";
    // materializeJS.src = "https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/js/materialize.min.js";
    script.type = "text/javascript";
    script.src = chrome.extension.getURL(src);
    (document.body || document.head || document.documentElement).appendChild(script);
    (document.body || document.head || document.documentElement).appendChild(materialize);
    (document.body || document.head || document.documentElement).appendChild(icons);
    // (document.body || document.head || document.documentElement).appendChild(materializeJS);
    script.onload = function(){

        var url= chrome.runtime.getURL("stopwords-es.txt");
        var evt=document.createEvent("CustomEvent");
        evt.initCustomEvent("yourCustomEvent", true, true, url);//mandamos direccion de stopwords-es.txt
        document.dispatchEvent(evt);

        var url_modelo = chrome.runtime.getURL("modelo_entrenado_json.json");
        var evt2=document.createEvent("CustomEvent");
        evt2.initCustomEvent("getModelo_url", true, true, url_modelo);//mandamos direccion de modelo_entrenado_json.json
        document.dispatchEvent(evt2);

        var evt3=document.createEvent("CustomEvent");
        evt3.initCustomEvent("close", true, true, url_modelo);//mandamos direccion de modelo_entrenado_json.json
        document.dispatchEvent(evt3);


    };
}



addScript("dist/extension.js");






