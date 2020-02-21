"use strict";

function addScript(src) {
    const script = document.createElement("script");
    script.type = "text/javascript";
    script.src = chrome.extension.getURL(src);
    (document.body || document.head || document.documentElement).appendChild(script);
    script.onload = function(){

	  	var url= chrome.runtime.getURL("stopwords-es.txt");
		var evt=document.createEvent("CustomEvent");
		evt.initCustomEvent("yourCustomEvent", true, true, url);//mandamos direccion de stopwords-es.txt
		document.dispatchEvent(evt);

		var url_modelo = chrome.runtime.getURL("modelo_entrenado_json.json");
		var evt2=document.createEvent("CustomEvent");
		evt2.initCustomEvent("getModelo_url", true, true, url_modelo);//mandamos direccion de stopwords-es.txt
		document.dispatchEvent(evt2);
	};
}





addScript("dist/extension.js");





