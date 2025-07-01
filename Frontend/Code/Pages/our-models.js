// Page with 5 buttons - Button Page Code

import wixLocation from 'wix-location';

$w.onReady(function () {

    // Button click handlers - redirect to chat page with model parameter
    $w("#tryAlpha").onClick(() => {
        wixLocation.to("/chat?model=alpha");
    });

    $w("#tryBeta").onClick(() => {
        wixLocation.to("/chat?model=beta");
    });

    $w("#tryGamma").onClick(() => {
        wixLocation.to("/chat?model=gamma");
    });

    $w("#tryDelta").onClick(() => {
        wixLocation.to("/chat?model=delta");
    });

    $w("#tryZero").onClick(() => {
        wixLocation.to("/chat?model=zero");
    });

});