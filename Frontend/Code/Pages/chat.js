// Frontend Page Code (e.g., Chat page - c1dmp.js)

import { initiate_guru_request, guru_request_status } from 'backend/asyncBot.jsw';
import wixWindow from 'wix-window';
import wixLocation from 'wix-location';
import { authentication, currentMember } from 'wix-members';

// --- Global Variables ---
let chatData = [];
const colorOptions = {
    red: "#FF0000",
    green: "#00FF00",
    blue: "#0000FF",
    yellow: "#FFFF00",
    cyan: "#00FFFF",
    magenta: "#FF00FF",
    black: "#000000",
    white: "#FFFFFF",
    gray: "#C0C0C0",
    orange: "#FFA500",
    purple: "#800080",
    brown: "#8B4513",
    darkBrown: "#5D2F0C",
    statusBg: "rgba(255, 255, 255, 0.15)"
};
let first_message = true;
let typing = false;
const POLLING_INTERVAL_MS = 3000;
let pollingIntervalId = null;
let requestStartTime = null;
let dotsAnimationInterval = null;
let currentStatusElement = null;
let lastStatusText = null;
let currentModel = null; // Track current model
let currentStatus = null; // Track current status to avoid re-typing
let lastStatusChangeTime = 0; // Track when status last changed
const MIN_STATUS_DISPLAY_TIME = 2000; // Minimum time to display each status (2 seconds)

// --- Page Ready Function ---

$w.onReady(function () {

    setUserIcon()

    // --------- set model's parameter -------------------------------------------------------------------

    setModelFromParameter();
    
    // --------- stop polling on re-rendering -------------------------------------------------------------------

    stopPolling();

    // --------- Click event -------------------------------------------------------------------

    $w("#buttonSend").onClick(() => {
        // Only proceed if not currently typing out a response AND input has text
        if (!typing && $w("#inputBox").value.trim()) {
            fade_intro_out(); // Optional: Fade out intro elements
            handle_user_message_async(); // Handle the message sending flow asynchronously
        }
    });

    // --------- Enter event -------------------------------------------------------------------

    $w('#inputBox').onKeyPress((event) => {
        
        if (event.key === "Enter" && !typing && $w("#inputBox").value.trim()) {
            fade_intro_out(); // Optional: Fade out intro elements
            handle_user_message_async(); // Handle the message sending flow asynchronously
        }
    });

});

async function setUserIcon() {

    let member = await currentMember.getMember();
    
    if (member) {
        $w("#userIcon").src = member.profile.profilePhoto.url;
    } 
}

// --- Format elapsed time helper function ---
function formatElapsedTime(seconds) {
    const totalSeconds = Math.round(seconds);
    
    if (totalSeconds < 60) {
        return `${totalSeconds} seconds`;
    } else {
        const minutes = Math.floor(totalSeconds / 60);
        const remainingSeconds = totalSeconds % 60;
        if (minutes === 1 && remainingSeconds === 0) {
            return `1 minute`;
        } else if (minutes === 1) {
            return `1 minute ${remainingSeconds} second${remainingSeconds !== 1 ? 's' : ''}`;
        } else if (remainingSeconds === 0) {
            return `${minutes} minutes`;
        } else {
            return `${minutes} minutes ${remainingSeconds} second${remainingSeconds !== 1 ? 's' : ''}`;
        }
    }
}

// --- Core Async Message Handling Logic ---

async function handle_user_message_async() {

    let model_value = $w("#model").value
    currentModel = model_value; // Store current model
    const model =   model_value === "zero" ? "URL_sciguru_zero" :
                    model_value === "alpha" ? "URL_sciguru_alpha" :
                    model_value === "beta" ? "URL_sciguru_beta" :
                    model_value === "gamma" ? "URL_sciguru_gamma" :
                    model_value === "delta" ? "URL_sciguru_delta" :
                    model_value === "phi" ? "URL_phi" : "URL_sciguru_zero"

    const userMessage = $w("#inputBox").value.trim();
    if (!userMessage) return;

    stopPolling();
    currentStatus = null; // Reset status tracking
    lastStatusChangeTime = 0; // Reset timing

    chatData.push({ sender: "user", text: userMessage });
    $w("#inputBox").value = "";

    
    // --------- Reset Chat -------------------------------------------------------------------

    const animationPromise = first_message ? Promise.resolve() : reset_chat_for_new_message();
    let currentTaskId = null;

    animationPromise
    
    // --------- Type user request -------------------------------------------------------------------

        .then(() => {
            $w("#userText").html = ""; // Clear previous user text (use .html)
            $w("#guruText").html = ""; // Clear previous guru text (use .html)

            // FIX #2: Ensure userChat shows before guruChat
            // First show the chatBox
            return $w("#chatBox").show("fade", { duration: 300 });
        })
        .then(() => {
            // Then show userChat and type the message
            return $w("#userChat").show("fade", { duration: 300 });
        })
        .then(() => {
            // Type the user message
            return update_user(userMessage);
        })

    // --------- Send request to the model -------------------------------------------------------------------

        .then(() => {
             console.log("Frontend: Initiating async bot request for : ", userMessage);
             // FIX #2: Show guruChat AFTER user message is typed
             return new Promise((resolve) => {
                 setTimeout(() => {
                     $w("#guruChat").show("fade", {duration: 300}).then(() => {
                         requestStartTime = Date.now(); // Start timing
                         update_guru("sending request", true, false, true); // Enable dots animation
                         resolve();
                     });
                 }, 100); // Small delay to ensure proper sequencing
             });
        })
        .then(() => {
             return initiate_guru_request(userMessage, model);
        })

    // --------- Type guru's response -------------------------------------------------------------------

        .then((taskId) => {
             console.log("Frontend: Received Task ID (", taskId, ")");
             if (!taskId) {
                 throw new Error("Backend did not return a valid Task ID.");
             }
             currentTaskId = taskId;
             update_guru("request sent successfully", true);
             startPolling(taskId);
        })

    // --------- Handle Errors-------------------------------------------------------------------

        .catch((error) => {
            console.error("Frontend Error during initiation phase:", error);
            $w("#guruChat").show("fade", {duration: 300});
            type_text($w("#guruText"), `Error starting request: ${error.message || 'Unknown error.'}`, 30, "Syne", "14px", colorOptions.red);
            stopPolling();
        })

    // --------- Update first message (its not the first one no more) -------------------------------------------------------------------

        .finally(() => {
             first_message = false;
        });
}

// --- Polling Functions ---

function startPolling(taskId) {
    console.log(`Frontend: Starting polling for Task ID: ${taskId}`);
    stopPolling();
    update_guru("checking model status", true, false, true); // Enable dots animation
    pollStatus(taskId); // Initial call
    pollingIntervalId = setInterval(() => { pollStatus(taskId); }, POLLING_INTERVAL_MS);
}

function stopPolling() {
    if (pollingIntervalId) {
        console.log("Frontend: Stopping polling.");
        clearInterval(pollingIntervalId);
        pollingIntervalId = null;
    }
    stopDotsAnimation();
}

// Dots animation functions
function startDotsAnimation(baseText) {
    stopDotsAnimation(); // Clear any existing animation
    let dots = "."; // Start with one dot
    
    console.log("Starting dots animation for:", baseText);
    
    // Immediately show the first dot state
    updateStatusWithDots(baseText, dots);
    
    dotsAnimationInterval = setInterval(() => {
        // Cycle through 1, 2, 3 dots
        if (dots === ".") {
            dots = "..";
        } else if (dots === "..") {
            dots = "...";
        } else {
            dots = ".";
        }
        console.log("Updating dots to:", dots);
        updateStatusWithDots(baseText, dots);
    }, 500); // Update every 500ms
}

function stopDotsAnimation() {
    if (dotsAnimationInterval) {
        clearInterval(dotsAnimationInterval);
        dotsAnimationInterval = null;
    }
    lastStatusText = null; // Reset last status text when stopping animation
}

function updateStatusWithDots(baseText, dots) {
    if (currentStatusElement && currentStatusElement.html) {
        // Recreate the whole status box with updated dots
        const styledText = `<span style="background: ${colorOptions.statusBg}; padding: 8px 12px; border-radius: 8px; display: inline-block;">
            <span style="color: ${colorOptions.darkBrown}; font-family: Syne; font-size: 14px; font-weight: bold;">${baseText} ${dots}</span>
        </span>`;
        currentStatusElement.html = `<div style="font-family: Syne; font-size: 14px; color: ${colorOptions.darkBrown}; line-height: 1.6; text-align: left;">${styledText}</div>`;
    }
}

async function pollStatus(taskId) {
    if (!pollingIntervalId) {
        console.log(`Frontend: Polling already stopped for Task ID: ${taskId}. Exiting pollStatus.`);
        return;
    }

    console.log(`Frontend: Polling status for Task ID: ${taskId}`);
    try {

        const statusResult = await guru_request_status(taskId);
        
        console.log("Frontend: Received status from backend:", statusResult);

        if (!pollingIntervalId) { // Check again if stopped during await
            console.log(`Frontend: Polling stopped while awaiting status for Task ID: ${taskId}. Ignoring result.`);
            return;
        }

        // Calculate time since last status change
        const timeSinceLastChange = Date.now() - lastStatusChangeTime;
        
        // Only update UI if status has changed AND minimum display time has passed
        if (statusResult.status !== currentStatus && timeSinceLastChange >= MIN_STATUS_DISPLAY_TIME) {
            const previousStatus = currentStatus;
            currentStatus = statusResult.status;
            lastStatusChangeTime = Date.now();
            
            console.log(`Frontend: Status changed from ${previousStatus} to ${currentStatus}`);
            
            switch (statusResult.status) {
                case "pending":
                    update_guru("request pending", true, false, true); // Enable dots animation
                    break;
                case "initializing":
                    // Show model-specific initialization message
                    const modelName = currentModel ? `SciGuru ${currentModel}` : "Model";
                    update_guru(`${modelName} initializing`, true, false, true); // Enable dots animation
                    break;
                case "running":
                    // Always show "thinking" message when we transition to running
                    update_guru("SciGuru is thinking", true, false, true); // Enable dots animation
                    break;
                case "complete":
                    stopPolling();
                    console.log("Frontend: Received complete result:", statusResult.result);
                    chatData.push({ sender: "bot", text: statusResult.result });
                    
                    // FIX #1: Calculate elapsed time with new format
                    const elapsedTimeSeconds = requestStartTime ? ((Date.now() - requestStartTime) / 1000) : null;
                    const formattedTime = elapsedTimeSeconds ? formatElapsedTime(elapsedTimeSeconds) : "unknown";
                    
                    // Create response with timing info in status box, then the actual response
                    const statusBox = `<span style="background: ${colorOptions.statusBg}; padding: 8px 12px; border-radius: 8px; display: inline-block;">
                        <span style="color: ${colorOptions.darkBrown}; font-family: Syne; font-size: 13px; font-weight: bold;">Response received in <span style="color: ${colorOptions.black}; font-weight: bold;">${formattedTime}</span></span>
                    </span>`;
                    
                    const responseWithTiming = `${statusBox}\n\n${statusResult.result}`;
                    
                    console.log("Setting HTML directly with timing. Length:", responseWithTiming.length);
                    update_guru(responseWithTiming, false, true); // true flag for raw HTML
                    break;
                case "error":
                    stopPolling();
                    update_guru("", true);
                    console.error("Frontend: Received error status from backend:", statusResult.error);
                    update_guru(`Sorry, an error occurred: ${statusResult.error || 'Unknown processing error.'}`, true)
                    break;
                default:
                    stopPolling();
                    update_guru("", true);
                    console.error("Frontend: Received unexpected status:", statusResult);
                    type_text($w("#guruText"), "Sorry, an unexpected response was received from the server.", 30, "Syne", "14px", colorOptions.red);
                    break;
            }
        }

    } catch (error) {
        console.error("Frontend Error during polling request:", error);
        stopPolling();
        update_guru("", true);
        type_text($w("#guruText"), "Error checking model status. Please try sending your message again.", 30, "Syne", "14px", colorOptions.red);
    }
}

// --- UI Helper Functions ---

function update_guru(text, status=false, isRawHtml=false, animateDots=false) {
    console.log("update_guru called with:", text, "status:", status, "isRawHtml:", isRawHtml, "animateDots:", animateDots);
    console.log("text length:", text?.length);

    let color = status ? colorOptions.darkBrown : colorOptions.white;
    let speed = status ? 30 : 7;

    // If dots animation is running and we're being called with the same status text, 
    // just return to preserve the animation
    if (animateDots && status && lastStatusText === text && dotsAnimationInterval) {
        console.log("Same status with dots animation already running, skipping update");
        return;
    }

    // Stop any existing dots animation only if we're changing the text
    if (!animateDots || lastStatusText !== text) {
        stopDotsAnimation();
    }

    if (!typing) {
        currentStatusElement = $w("#guruText"); // Keep track of element
        lastStatusText = text; // Track the last status text
        
        if (animateDots && status) {
            // For animated status messages, first type the text with "."
            const styledText = `<span style="background: ${colorOptions.statusBg}; padding: 8px 12px; border-radius: 8px; display: inline-block;">
                <span style="color: ${colorOptions.darkBrown}; font-family: Syne; font-size: 14px; font-weight: bold;">${text} .</span>
            </span>`;
            
            // Type the initial status with dots
            type_text_with_callback($w("#guruText"), styledText, speed, "Syne", "14px", color, true, () => {
                // After typing is complete, start the dots animation
                startDotsAnimation(text);
            });
        } else if (isRawHtml) {
            // For raw HTML (like response with timing), type it differently
            type_html_with_status($w("#guruText"), text, speed, "Syne", "14px", color);
        } else if (status) {
            // For static status messages, wrap in styled box
            const styledText = `<span style="background: ${colorOptions.statusBg}; padding: 8px 12px; border-radius: 8px; display: inline-block;">
                <span style="color: ${colorOptions.darkBrown}; font-family: Syne; font-size: 14px; font-weight: bold;">${text}</span>
            </span>`;
            type_text($w("#guruText"), styledText, speed, "Syne", "14px", color, true);
        } else {
            // Regular response text
            type_text($w("#guruText"), text, speed, "Syne", "14px", color);
        }
    }
}

function update_user(text) {
    return type_text($w("#userText"), text, 7, "Syne", "14px", colorOptions.white);
}

function fade_intro_out() {
  if (first_message) {
      $w("#cloudGinie")?.hide("fade", { duration: 500 });
      $w("#introGinie1")?.hide("slide", { duration: 200, direction: "right" })
        .then(() => $w("#introGinie2")?.hide("slide", { duration: 200, direction: "left" }))
        .then(() => $w("#introGinie3")?.hide("slide", { duration: 200, direction: "right" }));
  }
}

// FIX #3: Updated reset function to properly handle element references
function reset_chat_for_new_message() {
    return new Promise((resolve) => {
        // First hide the entire chatBox
        $w("#chatBox").hide("fade", { duration: 200 })
            .then(() => {
                // Clear the HTML content
                $w("#userText").html = "";
                $w("#guruText").html = "";
                
                // Hide individual chat containers
                $w("#userChat").hide();
                $w("#guruChat").hide();
                
                // Small delay to ensure clean transition
                setTimeout(() => {
                    resolve();
                }, 100);
            })
            .catch((error) => {
                console.error("Error in reset_chat_for_new_message:", error);
                resolve(); // Resolve anyway to continue the flow
            });
    });
}

// --- markdown + typing helper ---------------------------------------------

// Simple markdown parser that works with Wix Rich Text
function parseMarkdownToHTML(text, fontFamily, fontSize, color) {
    console.log("Parsing markdown to HTML:", text);
    console.log("Text length:", text.length);
    
    // First, let's see what we're actually getting
    if (!text || text.trim() === '') {
        console.log("Empty or null text received");
        return `<p style="margin: 4px 0; font-family: ${fontFamily}; color: ${color}; text-align: left;">No content</p>`;
    }
    
    // Calculate larger regular text size
    const baseFontSize = parseInt(fontSize) || 10;
    const regularTextSize = baseFontSize + 2; // Increase by 2px
    
    // Define lighter shades for headers (lighter than regular text)
    const colors = {
        h1: 'rgba(255, 255, 255, 0.9)', // 90% opacity - lightest
        h2: 'rgba(255, 255, 255, 0.85)', // 85% opacity
        reg: 'rgba(255, 255, 255, 0.75)'  // 80% opacity
    };
    
    let lines = text.split('\n');
    console.log("Split into lines:", lines);
    let htmlLines = [];
    
    for (let i = 0; i < lines.length; i++) {
        let line = lines[i];
        console.log(`Processing line ${i}: "${line}"`);

        // Skip empty lines but preserve spacing
        if (line.trim() === '') {
            htmlLines.push('<div style="height: 8px;"></div>');
            continue;
        }
        
        // Process headers - check for # at the very beginning of the line
        if (line.match(/^####\s/)) {
            htmlLines.push(`<h4 style="font-family: Syne; font-size: ${baseFontSize + 4}px; font-weight: bold; margin: 12px 0 8px 0; color: ${colors.h1}; text-align: left;">${line.substring(5).trim()}</h4>`);
        } else if (line.match(/^###\s/)) {
            htmlLines.push(`<h3 style="font-family: Syne; font-size: ${baseFontSize + 6}px; font-weight: bold; margin: 12px 0 8px 0; color: ${colors.h2}; text-align: left;">${line.substring(4).trim()}</h3>`);
        } else if (line.match(/^##\s/)) {
            htmlLines.push(`<h2 style="font-family: Syne; font-size: ${baseFontSize + 8}px; font-weight: bold; margin: 14px 0 10px 0; color: ${colors.h2}; text-align: left;">${line.substring(3).trim()}</h2>`);
        } else if (line.match(/^#\s/)) {
            htmlLines.push(`<h1 style="font-family: Syne; font-size: ${baseFontSize + 12}px; font-weight: bold; margin: 16px 0 12px 0; color: ${colors.h1}; text-align: left;">${line.substring(2).trim()}</h1>`);
        } else {
            // Process inline formatting
            let processedLine = line
                .replace(/\*\*(.*?)\*\*/g, `<strong>$1</strong>`)
                .replace(/\*(.*?)\*/g, `<em>$1</em>`)
                .replace(/`(.*?)`/g, `<code style="background: rgba(255,255,255,0.1); padding: 2px 4px; border-radius: 3px; font-family: monospace;">$1</code>`);
            
            htmlLines.push(`<p style="margin: 4px 0; font-family: Syne; font-size: ${regularTextSize}px; color: ${colors.reg}; text-align: left;">${processedLine}</p>`);
        }
    }
    
    let result = htmlLines.join('');
    console.log("Final HTML result:", result);
    return result;
}

export async function type_text_with_callback(element, rawText, speed, fontFamily, fontSize, color, isPreformatted = false, callback = null) {
    if (!element || typeof element.html === 'undefined') {
        console.error('type_text_with_callback: element missing or no .html support');
        return Promise.resolve(); // Return resolved promise instead of undefined
    }

    return new Promise((resolve) => {
        let htmlFull;
        if (isPreformatted) {
            // If pre-formatted (like status messages), use as-is
            htmlFull = rawText;
        } else {
            // Convert Markdown → HTML
            htmlFull = parseMarkdownToHTML(String(rawText || ''), fontFamily, fontSize, color);
        }
        
        const tokens = htmlFull.match(/<[^>]+>|./g) || [];

        typing = true;
        element.html = '';
        let built = '';
        let i = 0;

        (function step() {
            if (!typing) {
                resolve();
                return;
            }
            if (i >= tokens.length) {
                typing = false;
                if (callback) callback(); // Call callback when done
                resolve();
                return;
            }

            built += tokens[i++];
            element.html = `<div style="font-family: ${fontFamily}; font-size: ${fontSize}; color: ${color}; line-height: 1.6; text-align: left;">${built}</div>`;

            setTimeout(step, speed);
        })();
    });
}

export async function type_text(element, rawText, speed, fontFamily, fontSize, color, isPreformatted = false) {
    return type_text_with_callback(element, rawText, speed, fontFamily, fontSize, color, isPreformatted, null);
}

// Special function for typing HTML that contains both status box and regular content
async function type_html_with_status(element, htmlContent, speed, fontFamily, fontSize, color) {
    if (!element || typeof element.html === 'undefined') {
        console.error('type_html_with_status: element missing or no .html support');
        return;
    }

    // Split the content to find the status box and the regular content
    const statusBoxMatch = htmlContent.match(/(<span style="background:.*?<\/span>)/s);
    let statusBox = '';
    let regularContent = htmlContent;
    
    if (statusBoxMatch) {
        statusBox = statusBoxMatch[0];
        // Get content after the status box, skipping any newlines
        regularContent = htmlContent.substring(statusBoxMatch.index + statusBoxMatch[0].length).replace(/^\n+/, '');
    }

    typing = true;
    element.html = '';

    // First, type out the status box
    if (statusBox) {
        const statusTokens = statusBox.match(/<[^>]+>|./g) || [];
        let statusBuilt = '';
        
        for (let i = 0; i < statusTokens.length; i++) {
            statusBuilt += statusTokens[i];
            element.html = `<div style="font-family: ${fontFamily}; font-size: ${fontSize}; color: ${color}; line-height: 1.6; text-align: left;">${statusBuilt}</div>`;
            await new Promise(resolve => setTimeout(resolve, 20)); // Type status faster
        }
        
        await new Promise(resolve => setTimeout(resolve, 500)); // Brief pause after status
    }

    // Then type out the regular content
    const htmlFull = parseMarkdownToHTML(String(regularContent || ''), fontFamily, fontSize, color);
    const tokens = htmlFull.match(/<[^>]+>|./g) || [];
    
    let built = statusBox + '<div style="margin-top: 12px;"></div>'; // Add spacing after status
    let i = 0;

    (function step() {
        if (!typing) return;
        if (i >= tokens.length) {
            typing = false;
            return;
        }

        built += tokens[i++];
        element.html = `<div style="font-family: ${fontFamily}; font-size: ${fontSize}; color: ${color}; line-height: 1.6; text-align: left;">${built}</div>`;

        setTimeout(step, speed);
    })();
}

// Function to set model dropdown based on URL parameter
function setModelFromParameter() {
    try {
        // Get URL parameters
        const query = wixLocation.query;
        const modelParam = query.model;
        
        if (modelParam) {
            
            console.log("Setting model from URL parameter:", modelParam);            
            const dropdownValue = modelParam;
            if (dropdownValue) {
                $w("#model").value = dropdownValue;
                console.log("Model dropdown set to:", dropdownValue);
            } else {
                console.warn("Unknown model parameter:", modelParam);
            }
        }
    } catch (error) {
        console.error("Error setting model from parameter:", error);
    }
}