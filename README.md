# InSight
Building a voice-enabled ride along system for vision-impaired people to gain context, understanding, and awareness of the world around them.

## Background
Imagine you are on a busy street in New York. You need to meet your friend one block north and one block east. You begin walking down the street - you avoid the people coming from the left and right, dodge the many trash bins and fire hydrants in your path, and reach your friend frictionless.

Now do the same with your eyes closed. Walking forward is not as simple, as you might drift into the road. Avoiding hazards is difficult as well - you can't see the people intersecting you nor the many hazards in your way. There is no one to tell you what is around you, how much progress you have made, nor if you are walking into danger. The gift of information you originally had to get to your destination is gone.

Over 40 million across the world are blind. Over 250 million are severely vision impaired. Without access to a guide or vision correction, navigating the world becomes near impossible. Canes and guide dogs can provide direction or response, but provide no context into the surrounding environment and become difficult when used in spaces not fit for them. As a result, we built InSight to provide a pair of eyes to those without vision.

InSight is similar to a guide - if it senses any hazards, objects, or terrain to look out for, it'll play a warning sound then dictate into your phone speaker/headphones what it sees. It'll give directional input (left, right, in front) to try and create this sense of situational awareness. If there is need for any questions on the surrounding environment, enable listening mode to ask questions and get direct responses. InSight has a built in GPS to allow for smooth navigation to a destination and syncs with apple watch to allow for toggling the UI there.

The UI is built to be simplistic and easy-to-use for someone vision impaired. Simply tapping on the screen will enable the app. Swiping upp enables listening mode, swiping down enables automatic hazard detection mode. The UI works the same for the watch fixture.

Future plans for InSight are to give the models memory - understanding where the user has been before and what objects and features are persistent/new per stage. The ultimate goal with InSight is to be the eyes for millions around the world and ultimately bring the world around us in sight to those who can't see it.

## Tech Stack
- Python
- Swift
- Google Gemini API
- Google Direction and Geocoding API
- ElevenLabs API

## Backend Pipeline
- Every 5 seconds or so, a new image is taken from the live video feed.
- The image is added to a temporary cache to store the most recent captures.
- The image is fed to a similarity checker which checks for image similarity on 3 different fronts (pixel similarity, inlier similarity, and similarity by axis shift) between the new image and the most recent cache image besides the new one. If the image is deemed too similar on all 3 fronts, the image does not continue through the pipeline (as its information is now redundant from the last capture).
- If the image is new, it is passed through Gemini API to determine key hazards and features in the image. The output will be compared to a preset dictionary of hazards and severities and will fit into a stack rank of priority phrases to relay to the user. The phrases will then be relayed to the ElevenLabs API to then output the phrases into the speaker of the phone (or headphones if the user is wearing it).
- For navigation, Direction and Geocoding API are used to capture location of the mobile device and then calculate a set of steps to walk from one location to the other.


