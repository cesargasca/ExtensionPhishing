var gmail;


function refresh(f) {
  if( (/in/.test(document.readyState)) || (typeof Gmail === undefined) ) {
    setTimeout('refresh(' + f + ')', 10);
  } else {
    f();
  }
}


var main = function(){
  	// NOTE: Always use the latest version of gmail.js from
 	// https://github.com/KartikTalwar/gmail.js
  	//gmail = new Gmail();
  	var classifier = new aja.BayesClassifier();
  	/*console.log('Hello,', gmail.get.user_email())
  	var email_data = gmail.get.thread_id() 
	console.log(email_data)
  
  
	var email = new gmail.dom.email(email_data); // optionally can pass relevant $('div.adn');
	var body = email.body();
	var id = email.id;

	add a heading at the start of the email and update in the interface
	email.body('<h1>My New Heading!</h1>' + body);
	console.log(body)
  	console.log(email_data.[""0""].innerHTML)*/


  	classifier.addDocument('my unit-tests failed.', 'software');
	classifier.addDocument('tried the program, but it was buggy.', 'software');
	classifier.addDocument('the drive has a 2TB capacity.', 'hardware');
	classifier.addDocument('i need a new power supply.', 'hardware');

	classifier.train();

	console.log(classifier.classify('did the tests pass?'));
	console.log(classifier.classify('did you buy a new drive?'));
}


refresh(main);
