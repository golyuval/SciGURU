// Frontend Page Code (e.g., Chat page - c1dmp.js)

// Velo API Reference: https://www.wix.com/velo/reference/api-overview/introduction
import { initiateBotRequest, getBotRequestStatus } from 'backend/asyncBot.jsw'; // Import NEW backend functions

// --- Global Variables ---
let chatData = []; // Array to store chat messages { sender: "user" | "bot", text: string }
const colorOptions = {
    red: "#FF0000", // Use for errors
    green: "#00FF00",
    blue: "#0000FF",
    yellow: "#FFFF00",
    cyan: "#00FFFF",
    magenta: "#FF00FF",
    black: "#000000",
    white: "#FFFFFF", // Use for user and successful bot text
    gray: "#808080", // Use for info/status messages
    orange: "#FFA500",
    purple: "#800080"
};
let first_message = true; // Flag to handle initial animations/messages
let typing = false; // Flag to indicate if the typing effect is active
const POLLING_INTERVAL_MS = 3000; // Poll status every 3 seconds
let pollingIntervalId = null; // Stores the ID of the setInterval timer

// --- Page Ready Function ---
$w.onReady(function () {
    // Stop any potentially orphaned polling intervals if the page reloads/re-renders
    stopPolling();

    // --- Event Handlers ---
    $w("#buttonSend").onClick(() => {
        // Only proceed if not currently typing out a response AND input has text
        if (!typing && $w("#inputBox").value.trim()) {
            fade_intro_out(); // Optional: Fade out intro elements
            handle_user_message_async(); // Handle the message sending flow asynchronously
        }
    });

    $w('#inputBox').onKeyPress((event) => {
        // Only proceed on Enter key, if not typing, AND input has text
        if (event.key === "Enter" && !typing && $w("#inputBox").value.trim()) {
            fade_intro_out(); // Optional: Fade out intro elements
            handle_user_message_async(); // Handle the message sending flow asynchronously
        }
    });

    // --- Initial Page Setup (Optional) ---
    // $w("#chatBox").hide();
    // $w("#userBox").hide();
    // $w("#guruBox").hide();
});

// --- Core Async Message Handling Logic ---

async function handle_user_message_async() {

    const userMessage = $w("#inputBox").value.trim();
    if (!userMessage) return;

    stopPolling();

    chatData.push({ sender: "user", text: userMessage });
    $w("#inputBox").value = "";

    const animationPromise = first_message ? Promise.resolve() : reset_chat_for_new_message();
    let currentTaskId = null;

    animationPromise
        .then(() => {
            $w("#userText").html = ""; // Clear previous user text (use .html)
            $w("#guruText").html = ""; // Clear previous guru text (use .html)
            $w("#chatBox").show("fade", { duration: 300 });
            $w("#userBox").show("fade", { duration: 300 });
            $w("#guruBox").hide();
            return typeText($w("#userText"), userMessage, 30, "Syne", "16px", colorOptions.white);
        })
        .then(() => {
             console.log("Frontend: Initiating async bot request for : ", userMessage);
             $w("#guruText").html = `<p style="font-family: Syne; font-size: 16px; color: ${colorOptions.gray}; margin:0; padding:0;">Sending request...</p>`; // Use HTML for consistency
             $w("#guruBox").show("fade", {duration: 300});
             return initiateBotRequest(userMessage);
        })
        .then((taskId) => {
             console.log("Frontend: Received Task ID : ", taskId);
             if (!taskId) {
                 throw new Error("Backend did not return a valid Task ID.");
             }
             currentTaskId = taskId;
             updateStatusText("Request sent. Waiting for model...");
             startPolling(taskId);
        })
        .catch((error) => {
            console.error("Frontend Error during initiation phase:", error);
            $w("#guruBox").show("fade", {duration: 300});
            typeText($w("#guruText"), `Error starting request: ${error.message || 'Unknown error.'}`, 30, "Syne", "16px", colorOptions.red);
            stopPolling();
        })
        .finally(() => {
             first_message = false;
        });
}

// --- Polling Functions ---

function startPolling(taskId) {
    console.log(`Frontend: Starting polling for Task ID: ${taskId}`);
    stopPolling();
    updateStatusText("Checking model status...");
    pollStatus(taskId); // Initial call
    pollingIntervalId = setInterval(() => { pollStatus(taskId); }, POLLING_INTERVAL_MS);
}

function stopPolling() {
    if (pollingIntervalId) {
        console.log("Frontend: Stopping polling.");
        clearInterval(pollingIntervalId);
        pollingIntervalId = null;
    }
}

async function pollStatus(taskId) {
    if (!pollingIntervalId) {
        console.log(`Frontend: Polling already stopped for Task ID: ${taskId}. Exiting pollStatus.`);
        return;
    }

    console.log(`Frontend: Polling status for Task ID: ${taskId}`);
    try {
        const statusResult = await getBotRequestStatus(taskId);
        console.log("Frontend: Received status from backend:", statusResult);

        if (!pollingIntervalId) { // Check again if stopped during await
            console.log(`Frontend: Polling stopped while awaiting status for Task ID: ${taskId}. Ignoring result.`);
            return;
        }

        switch (statusResult.status) {
            case "pending":
            case "running":
                updateStatusText("Model is processing your request...");
                break;
            case "complete":
                stopPolling();
                updateStatusText("");
                console.log("Frontend: Received complete result:", statusResult.result);
                chatData.push({ sender: "bot", text: statusResult.result });
                console.log("Setting HTML directly. Length:", statusResult.result.length); // Log length too
$w("#guruText").html = `<p style="font-family: Syne; font-size: 16px; color: ${colorOptions.white}; white-space: normal; word-break: break-word; margin:0; padding:0;">${statusResult.result.replace(/\n/g, '<br>')}</p>`; // Use .html and replace newlines
// OR try with .text if .html causes issues, though styling will be lost:
// $w("#guruText").text = statusResult.result;
                //typeText($w("#guruText"), statusResult.result, 30, "Syne", "16px", colorOptions.white);
                break;
            case "error":
                stopPolling();
                updateStatusText("");
                console.error("Frontend: Received error status from backend:", statusResult.error);
                typeText($w("#guruText"), `Sorry, an error occurred: ${statusResult.error || 'Unknown processing error.'}`, 30, "Syne", "16px", colorOptions.red);
                break;
            default:
                stopPolling();
                updateStatusText("");
                console.error("Frontend: Received unexpected status:", statusResult);
                typeText($w("#guruText"), "Sorry, an unexpected response was received from the server.", 30, "Syne", "16px", colorOptions.red);
                break;
        }

    } catch (error) {
        console.error("Frontend Error during polling request:", error);
        stopPolling();
        updateStatusText("");
        typeText($w("#guruText"), "Error checking model status. Please try sending your message again.", 30, "Syne", "16px", colorOptions.red);
    }
}

// --- UI Helper Functions ---

function updateStatusText(text) {
     if (!typing) {
         // Use .html consistently since typeText uses it
         $w("#guruText").html = `<p style="font-family: Syne; font-size: 16px; color: ${colorOptions.gray}; margin:0; padding:0;">${text}</p>`;
         $w("#guruBox").show();
     }
}

function fade_intro_out() {
  if (first_message) {
      $w("#cloudGinie")?.hide("fade", { duration: 500 });
      $w("#introGinie1")?.hide("slide", { duration: 200, direction: "right" })
        .then(() => $w("#introGinie2")?.hide("slide", { duration: 200, direction: "left" }))
        .then(() => $w("#introGinie3")?.hide("slide", { duration: 200, direction: "right" }));
  }
}

function reset_chat_for_new_message() {
  return $w("#chatBox")
      .hide("fade", { duration: 200, direction: "up" })
      .then(() => {
          $w("#userBox").hide();
          $w("#guruBox").hide();
          $w("#userText").html = "";
          $w("#guruText").html = "";
          return Promise.resolve();
      });
}

function animate(item, obj, type, delay, duration, direction) {
    return new Promise((resolve) => {
      setTimeout(() => {
        const element = item(obj);
        if (element && element.show) {
            element.show(type, { duration: duration, direction: direction });
        } else {
            console.warn(`Warning: Element "${obj}" not found or doesn't support .show() in animate function.`);
        }
        resolve();
      }, delay);
    });
}

// --- CORRECTED typeText Function ---
export function typeText(element, fullText, speed, fontFamily, fontSize, color) {
  return new Promise((resolve, reject) => {
     if (!element || typeof element.html === 'undefined') {
        console.error("Error: Invalid element passed to typeText (must support .html):", element);
        typing = false;
        return reject(new Error("Invalid element for typing effect."));
    }

    const textToType = String(fullText ?? "");

    if (typing) {
        console.warn("Warning: typeText called while already typing. Resetting.");
        // Consider adding logic to cancel previous interval if needed
    }
    typing = true;
    element.html = "";

    let typedHTML = "";
    let index = 0;

    function addChar() {
      if (!typing) {
          console.log("Typing stopped externally.");
          return resolve();
      }

      if (index < textToType.length) {
        const currentChar = textToType[index];

        // --- CORRECTED HTML ESCAPING ---
        if (currentChar === '<') { typedHTML += '<'; }
        else if (currentChar === '>') { typedHTML += '>'; }
        else if (currentChar === '&') { typedHTML += '&'; }
        // --- END CORRECTION ---
        else if (currentChar === '\n') { typedHTML += '<br>'; }
        // Use   entity for spaces that need preserving
        else if (currentChar === ' ' && (index === 0 || [' ', '\n'].includes(textToType[index-1]))) { typedHTML += ' '; }
        else { typedHTML += currentChar; }

        // Update the element's HTML
        element.html = `<p style="font-family: ${fontFamily}; font-size: ${fontSize}; color: ${color}; white-space: normal; word-break: break-word; margin:0; padding:0;">${typedHTML}</p>`;

        index++;
        setTimeout(addChar, speed);
      } else {
        typing = false;
        resolve();
      }
    }
    addChar();
  });
}

// Optional: Add functions to disable/enable input/button while processing
// function disableInput() { $w("#inputBox").disable(); $w("#buttonSend").disable(); }
// function enableInput() { $w("#inputBox").enable(); $w("#buttonSend").enable(); }