/* voices-for-agents.js

Experimental voice-synth for use in teaching and res. projects.

Voices available seem to be quite browser and platform dependent.

Reference:
https://developer.chrome.com/blog/web-apps-that-talk-introduction-to-the-speech-synthesis-api

*/
//alert("About to speak; if your browser is new to this page, you might want to enable sound in your site settings.");

function asyncSpeak(voice, text) {
    const ssu = new SpeechSynthesisUtterance()
      ssu.text = text
      if (voice==null) {
      ssu.lang = 'fr-FR'}
      else {
          ssu.voice = voice}
      speechSynthesis.speak(ssu);
      return new Promise((resolve, reject) => {
      ssu.onend = () => {
          resolve()
      }
    })
  }
  
  async function asyncTalk(voice, text, callback) {
      await asyncSpeak(voice, text);
      callback();    
  }
  
  const synth = speechSynthesis;
  
  let voices = [];
  
  function loadVoices() {
  // Fetch the available voices.
     voices = synth.getVoices();
  }
  
  function startVoices() {
      // This should only run if the GUI option has use-voices set to On.
      alert("startVoices has been called.");
      loadVoices();
      // Chrome loads voices asynchronously.
      window.speechSynthesis.addEventListener("voiceschanged", () => {
        loadVoices();
        window.theVoice = voices[1]; // Arbitrary default
    });
  }
  
  function detectBrowser() {
    const userAgent = navigator.userAgent;
  
    if (userAgent.match(/edg|microsoft/i)) {
      b = "Edge";
    } else if (userAgent.match(/chrome|chromium|crios/i)) {
      b = "Chrome";
    } else if (userAgent.match(/firefox|fxios/i)) {
      b = "Firefox";
    } else if (userAgent.match(/safari/i)) {
      b = "Safari";
    } else if (userAgent.match(/msie|trident/i)) {
      b = "Internet Explorer";
    } else if (userAgent.match(/opera|opr/i)) {
      b = "Opera";
    } else {
      b = "Unknown";
    }
      return b;
  }
  
  function detectOS() {
    const userAgent = navigator.userAgent;
    if (userAgent.match(/windows|Windows/i)) {
      os = "Windows";
    } else if (userAgent.match(/Mac OS/i)) {
      os = "Mac";
    } else if (userAgent.match(/linux|Linux/i)) {
      os = "Linux";
    } else if (userAgent.match(/android|Android|ANDROID/i)) {
      os = "Android";
    } else if (userAgent.match(/ios/i)) {
      os = "ios";
    } else {
      os = "Unknown";
    }
      return os
  }
  
  const browser = detectBrowser();
  if (browser=="Firefox") {alert("Browser is:" + browser); }
  
  const OS = detectOS();
  if (OS=="Mac") {alert("OS is:" + OS); }
  
  function getNarratorVoice () {
      // Look for a specific voice.
      // THESE COULD BE MADE OS DEPENDENT.
      // The choices here might not please you; feel free to change them.
      let voice_desc = "Default";
      
  
    if (browser=="Firefox") {
        voice_desc = "Microsoft Zira - English (United States)";
    }
    if (browser=="Edge") {
        voice_desc = "Microsoft Molly Online (Natural) - English (New Zealand)"; 
    }
    if (browser=="Chrome") {
        voice_desc = "Google Nederlands"; 
    }
      //console.log("Voice selected is: "+voice_desc);
      voice = find_voice(voice_desc);
  //   if (browser=="Firefox") {alert("In Firefox, voice found: "+voice_desc); }   
      return voice;
  }
  function getPlayerXVoice () {
    // Look for a specific voice.
    let voice_desc = "Default";
  
    if (browser=="Firefox") {
        voice_desc = "Microsoft Mark - English (United States)";
    }
    if (browser=="Edge") {
        voice_desc = "Microsoft Molly Online (Natural) - English (New Zealand)";
    }
    if (browser=="Chrome") {
        voice_desc = "Google UK English Female"; 
    }
      console.log("Voice selected is: "+voice_desc);
      voice = find_voice(voice_desc);
      return voice;
  }
  function getPlayerOVoice () {
    // Look for a specific voice.
    let voice_desc = "Default";
  
    if (browser=="Firefox") {
        voice_desc = "Microsoft Mark - English (United States)";
    }
    if (browser=="Edge") {
        voice_desc = "Microsoft Antoine Online (Natural) - French (Canada)"; // -
    }
    if (browser=="Chrome") {
        voice_desc = "Google UK English Male";
    }
      console.log("Voice selected is: "+voice_desc);
      voice = find_voice(voice_desc);
      return voice;
  }
  
  function getDefaultVoice() {
      return voices[0];
  }
  
  function find_voice(vdesc) {
      console.log("Number of voices: "+voices.length);    // Loop through each of the voices.
      voices.forEach(
      function(voice, i) {
          if (i==0) {theVoice = voice;}
          console.log(voice.name);
          if (voice.name==vdesc) {
          console.log("found voice: "+vdesc);
          window.theVoice = voice;
                  //talk(voice, "Yay! I found my voice.");
          }
      }
      );
      if (window.theVoice.name != vdesc) {
      alert("Could not find voice: ["+vdesc+"] on current platform (OS="+OS+", and browser="+browser+").");
      window.theVoice = null;
      console.log("Could not find voice: "+vdesc+" on current platform (OS="+OS+", and browser="+browser+").")
      }
      return window.theVoice;
  }
  