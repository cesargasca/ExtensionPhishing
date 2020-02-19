"use strict";

console.log("Extension loading...");
const jQuery = require("jquery");
const $ = jQuery;
const GmailFactory = require("gmail-js");
const gmail = new GmailFactory.Gmail($);
window.gmail = gmail;

gmail.observe.on("load", () => {
    const userEmail = gmail.get.user_email();
    console.log("Hello, " + userEmail + ". This is your extension talking!");

    gmail.observe.on("view_email", (domEmail) => {
        console.log("Looking at email:", domEmail);
        const emailData = gmail.new.get.email_data(domEmail);
        console.log("Email data:", emailData);
        console.log("TEST")
        var email_id = gmail.get.thread_id() 
    	var email_dom = new gmail.dom.email(email_id); // optionally can pass relevant $('div.adn');
		var body = email_dom.body();
		console.log(body)
	//add a heading at the start of the email and update in the interface
		
		
    });
});
