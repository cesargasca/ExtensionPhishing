"use strict";

console.log("Extension loading...");
const jQuery = require("jquery");
const $ = jQuery;
const GmailFactory = require("gmail-js");
//const lorca = require('lorca-nlp');
var natural = require('natural');
const gmail = new GmailFactory.Gmail($);
window.gmail = gmail;

function removeTags(str) {
      if ((str===null) || (str===''))
      return false;
      else
      str = str.toString();
      return str.replace( /(<([^>]+)>)/ig, '');
   }








gmail.observe.on("load", () => {
    const userEmail = gmail.get.user_email();
    console.log("Hello, " + userEmail + ". This is your extension talking!");

    gmail.observe.on("view_email", (domEmail) => {
        console.log("Looking at email:", domEmail);
        var email_id = gmail.get.thread_id() //get email id
    	var email_dom = new gmail.dom.email(email_id); // get dom
		var body = email_dom.body();
        var body_without_tags = removeTags(body)
        //var doc = lorca(body_without_tags);
        var tokenizer = new natural.WordTokenizer();
    console.log(tokenizer.tokenize(body_without_tags));

    });
});

