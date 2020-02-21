"use strict";

console.log("Extension loading...");
const jQuery = require("jquery");
const $ = jQuery;
const GmailFactory = require("gmail-js");
const lorca = require('lorca-nlp');
const gmail = new GmailFactory.Gmail($);
var sw = [];
var modelo_entrenado = "";
window.gmail = gmail;

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
        console.log(modelo_json);
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
      console.log(tokens[i]);
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
  console.log(p_p);
  console.log(p_h);

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

        var tokens = doc.uniqueWords().get(); //obtiene tokens unicos
        var tokens_sin_stopwords =  removeStopWords(tokens); //quita stopwords
        var cleaned_tokens = cleanTokens(tokens_sin_stopwords); //elimina tokens con caracteres especiales y con tamaño menor de 2

        //lemmatizar, stemming, considerar tfidf
        //probar modelo de clasificacion

        //mostrar al usuario
        
        console.log(tokens) 
        console.log(tokens_sin_stopwords);
        console.log(cleaned_tokens);

        //console.log(modelo_entrenado.prior_phishing);
        if(test(cleaned_tokens)=="phishing")
          email_dom.body('<h1 style=\"color:red\">PHISHING!!!!!!!!!!!!!!!!!!</h1>' + body);
        else
          email_dom.body('<h1 style=\"color:green\">NO PASA NADA</h1>' + body);
    });
});