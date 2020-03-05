"use strict";

console.log("Extension loading...");
const jQuery = require("jquery");
const $ = jQuery;
const GmailFactory = require("gmail-js");
const lorca = require('lorca-nlp');
const LanguageDetect = require('languagedetect');
const gmail = new GmailFactory.Gmail($);
var sw = [];
var modelo_entrenado = "";
window.gmail = gmail;
const lngDetector = new LanguageDetect();



const danger = "<div class='row' id='alert-id'>\
                <div class='col s6 m6'>\
                    <div id='card-alert' class='card red'>\
                      <div class='card-content white-text'>\
                          <span class='card-title white-text darken-1'>\
                            <i class='material-icons left'>warning</i>\
                            PHISHING\
                          </span>\
                        <p>\
                        Este correo electrónico parece tener intensiones de phishing\
                        </p>\
                      </div>\
                      <div class='card-action red darken-2'>\
                        <a id='close_row' class='btn-small waves-effect light-blue white-text'><i class='material-icons left'>check</i> Parece seguro</a>\
                        <a class='btn-small waves-effect red accent-2 white-text'><i class='material-icons left'>close</i> Mover a etiqueta phishing</a>\
                      </div>\
                    </div>\
                </div>\
                </div>";
               
const relax = "<div class='row' id='alert-id'>\
                <div class='col s6 m6'>\
                <div id='card-alert' class='card green'>\
                      <div class='card-content white-text'>\
                        <span class='card-title white-text darken-1'>\
                            <i class='material-icons left'>wb_sunny</i>\
                            NO PHISHING\
                          </span>\
                        <p>\
                        Este correo electrónico parece NO tener intensiones de phishing\
                        </p>\
                      </div>\
                      <div class='card-action green darken-2'>\
                        <a id='close_row' class='btn-small waves-effect light-blue white-text'><i class='material-icons left'>check</i> Ok</a>\
                        <a class='btn-flat waves-effect red accent-2 white-text'><i class='material-icons left'>close</i> Mover a etiqueta phishing</a>\
                      </div>\
                    </div>\
                </div>\
                </div>";

const noSpanish = "<div class='row' id='alert-id'>\
                <div class='col s6 m6'>\
                <div id='card-alert' class='card blue'>\
                      <div class='card-content white-text'>\
                        <span class='card-title white-text darken-1'>\
                            <i class='material-icons left'>language</i>\
                            Parece que este correo no está en español\
                      </div>\
                      <div class='card-action blue darken-2'>\
                        <a id='close_row' class='btn-small waves-effect light-green white-text'><i class='material-icons left'>check</i> Ok</a>\
                      </div>\
                    </div>\
                </div>\
                </div>";





function removeTags(str) {
      if ((str===null) || (str===''))
      return false;
      else
      str = str.toString();
      return str.replace( /(<([^>]+)>)/ig, '');
   }


document.addEventListener('yourCustomEvent', function (e)
{
    var url=e.detail;
    console.log("received "+url);
    var xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.onreadystatechange = function()
{
    if(xhr.readyState == XMLHttpRequest.DONE && xhr.status == 200)
    {
        var stopwords = xhr.responseText.split('\n');
        var cleaned = cleanStopWords(stopwords);
        sw = cleaned; //leemos stopwords y elimnina ultimo caracter basura

    }
};
xhr.send();
});

document.addEventListener('getModelo_url', function (e)
{
    var url=e.detail;
    console.log("received "+url);
    var xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.onreadystatechange = function()
{
    if(xhr.readyState == XMLHttpRequest.DONE && xhr.status == 200)
    {
        var modelo_json = xhr.responseText;
        //console.log(modelo_json);
        modelo_entrenado = JSON.parse(modelo_json)
    }
};
xhr.send();
});


function removeStopWords(tokens){
  var tokens_sin_stopwords = []

  for(var i = 0; i < tokens.length ; i++){
    if(sw.includes(tokens[i]) == false){
      tokens_sin_stopwords.push(tokens[i]);
    }
    else{
      console.log(tokens[i]);
    }
  }
  return tokens_sin_stopwords;
}

function cleanStopWords(stopwords){
  var cleaned = []
    for(var i= 0; i < stopwords.length ; i++){
      var aux = stopwords[i].substring(0,stopwords[i].length-1);
      //console.log(stopwords[i].length,aux.length);
        cleaned.push(aux);
    }
    return cleaned;

}



function cleanTokens(tokens){
  var cleaned = [];
  var format = /[ !@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/;
  for(var i = 0 ; i < tokens.length ; i++){
    if(tokens[i].length <= 2 || format.test(tokens[i])){
      //console.log(tokens[i]);
    }
    else{
      cleaned.push(tokens[i]);
    }
  }

  return cleaned
}



function test(tokens){
  var p_p = modelo_entrenado.prior_phishing;
  var p_h = modelo_entrenado.prior_ham;
  //console.log(p_p);
  //console.log(p_h);

  var conditional_probability_phishing = 1;
    var conditional_probability_ham = 1;

  for(var i = 0; i < tokens.length ; i++){
    var probability = modelo_entrenado.p_phishing[tokens[i]]
    var probability_h = modelo_entrenado.p_ham[tokens[i]]
    if(probability != null && parseFloat(probability) > 0){
      console.log("phishing",tokens[i],probability);
      conditional_probability_phishing *= probability;
    }
    if(probability_h != null && parseFloat(probability_h) > 0){
      conditional_probability_ham *= probability_h;
      console.log("ham",tokens[i],probability_h)
    }
  }
  
    var result_phishing = p_p*conditional_probability_phishing
    var result_ham = p_h*conditional_probability_ham
    console.log(result_phishing)
    console.log(result_ham)

    if(result_phishing > result_ham)
        return "phishing";
    else
        return "ham";
}


gmail.observe.on("load", () => {
    const userEmail = gmail.get.user_email();
    console.log("Hello, " + userEmail + ". This is your extension talking!");

    gmail.observe.on("view_email", (domEmail) => {
        console.log("Looking at email:", domEmail);
        var email_id = gmail.get.thread_id() //get email id
      var email_dom = new gmail.dom.email(email_id); // get dom
    var body = email_dom.body();
        var body_without_tags = removeTags(body)//elimina etiquetas html
        var doc = lorca(body_without_tags);
        console.log(doc)
        var languajes = lngDetector.detect(body_without_tags,2);
  
        if(languajes[0][0] == "spanish" || languajes[0][1] == "spanish"){
            //var tokens = doc.uniqueWords().get(); //obtiene tokens unicos
            var tokens_stemmed = doc.words().stem().get() //obtiene tokens ya con stemming
            var tokens_sin_stopwords =  removeStopWords(tokens_stemmed); //quita stopwords
            var cleaned_tokens = cleanTokens(tokens_sin_stopwords); //elimina tokens con caracteres especiales y con tamaño menor de 2
            //console.log(tokens); //imprime tokens
            console.log(tokens_stemmed);
            console.log(tokens_sin_stopwords);
            console.log(cleaned_tokens);
            var set = new Set(cleaned_tokens);
            console.log(Array.from(set));
            if(test(Array.from(set))=="phishing"){

                email_dom.body(danger + body);
                document.getElementById("close_row").addEventListener("click", function cierra() {
                    document.getElementById("alert-id").style.display = "none";
                });
            }
            else{
                email_dom.body( relax + body);
                 document.getElementById("close_row").addEventListener("click", function cierra() {
                    document.getElementById("alert-id").style.display = "none";
                });
            }
        }
        else{
            email_dom.body(noSpanish + body);
            document.getElementById("close_row").addEventListener("click", function cierra() {
                    document.getElementById("alert-id").style.display = "none";
                });

        }
             

    });
});