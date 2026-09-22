# 22 Sep 2026

This pivot clarifies the **User Experience (UX)** significantly. You are moving from a "Content Creation Tool" (like a movie studio) to a **"Content Consumption Platform"** (like a Netflix for literature/spirituality).

The goal is **Instant Transformation:** The user shouldn't feel like they are "making a movie"; they should feel like they are "watching a movie version of a book" that happens to feature their friends/family.

Here is the updated set of assumptions and the refined prompt to move forward.

### Part 1: New App Name Ideas (Consumption Focused)
1.  **Visionary** (Simple, elegant, focuses on "seeing" the words)
2.  **LuminaBooks** (Light/Vision + Books)
3.  **AuraScript** (Good for spiritual/philosophical texts)
4.  **KineticRead** (The idea of "moving" books)

---

### Part 2: Updated Assumptions (Fine-tuned for "Video Books")
1.  **On-Demand Generation:** We assume the video isn't pre-rendered for everyone. It is generated *the moment* a user clicks "Watch," or is cached for the first person who watches it.
2.  **Dynamic Casting:** We assume the "Actor" selection is a global setting. If I choose my friend to play "Krishna," he should appear in every chapter of the video book I watch.
3.  **Narrative Consistency:** Since the user isn't "directing," the AI must have a "Director's Brain"—it must decide the camera angles, the lighting, and the pacing based on the book's mood.
4.  **Latency Management:** We assume that because "Real-time" video generation is heavy, the app might need a "Generating your chapter..." loading state (similar to how some AI images take 10 seconds to load).

---

### Part 3: The Master Prompt (Iteration 2)
*Use this prompt to start the next phase of planning with an AI.*

> **Prompt:**
> "I am building a 'Video Book' platform. The core concept is an interactive library where users can select a book (e.g., the Bhagavad Gita) and click 'Watch' on any chapter. The app will instantly generate a cinematic video of that chapter.
>
> **Key Features:**
> 1. **Personalized Casting:** Users can upload photos of real people to act as the characters. The AI must ensure these specific faces are mapped onto the characters consistently throughout the book.
> 2. **Voice Cloning:** Users upload audio samples so that the characters speak in the voices of their chosen 'actors.'
> 3. **Automated Direction:** The user does NOT provide prompts for scenes. The AI must analyze the text, determine the setting (e.g., a battlefield, a forest, a palace), and generate the visuals, camera movements, and music automatically.
> 4. **Library Interface:** A clean UI where users can browse books, see chapter summaries, and toggle between 'Reading Mode' and 'Watching Mode.'
>
> **Your Task:**
> Act as a Product Architect. Please provide:
> 1. **The 'Engine' Logic:** Describe the pipeline of how a text chapter turns into a video. (Text $\rightarrow$ Script $\rightarrow$ Scene Descriptions $\rightarrow$ Video/Audio $\rightarrow$ Final Render).
> 2. **Consistency Strategy:** How can we ensure that 'Character A' looks the same in Chapter 1 as they do in Chapter 10? 
> 3. **User Flow:** Describe the onboarding process. How does a user 'set up' their cast before they start watching the books?
> 4. **Monetization & Scalability:** Since video generation is expensive, suggest a model (e.g., credits, subscription, or 'render once' sharing).
>
> **Note:** Do not write code. I want to understand the architectural flow and user experience first."

---

### Why this prompt works:
*   **"The Engine Logic":** This forces the AI to explain how it handles the "heavy lifting" of turning text into multiple video files.
*   **"Consistency Strategy":** This addresses your biggest technical hurdle (making sure the person doesn't change faces every 5 seconds).
*   **"Onboarding":** This helps you think about how to get a user to provide their photos/voices without it feeling like a chore.

# 21 Sep 2026 
This is an ambitious concept that sits at the intersection of **Generative AI (Video, Audio, and Image Synthesis)** and **Content Automation**.

To build this, we need to frame it as a "Personalized AI Cinematic Studio." Instead of just a "video maker," it is a tool that preserves identity (voice/face) while automating complex storytelling.

Here are the names, assumptions, and the initial prompt framework for your first iteration.

---

### Part 1: App Name Ideas
Since the app deals with "Scripting," "Identity," and "Visual Storytelling," here are 4 directions:

1.  **Viraaj AI** (Sanskrit for "Majestic/Brilliant" – fits the spiritual/epic theme)
2.  **PersonaCinema** (Focuses on the "Actors" and "Voice" aspect)
3.  **MythosFlow** (Focuses on the "Storytelling" and "Flow" of chapters)
4.  **ScriptLens** (Focuses on turning text/concepts into visual reality)

---

### Part 2: Initial Assumptions (To be fine-tuned)
Before building, we must assume how the technology behaves. We will test these in your next iterations:

1.  **Identity Persistence:** We assume the AI can maintain the same face and voice across different scenes (no "morphing" between shots).
2.  **Contextual Awareness:** We assume the AI understands the *tone* of the source material (e.g., the Gita is philosophical/epic, not a modern sitcom).
3.  **Sequential Consistency:** We assume the app can remember what happened in Chapter 8, Verse 1 so that Chapter 8, Verse 2 follows logically in the video.
4.  **User Input Depth:** We assume the user provides enough "vibe" instructions (e.g., "Cinematic, 4K, slow motion") to get high-quality results.
5.  **Voice Cloning:** We assume the user provides at least 30–60 seconds of clear audio for a "sample voice" to be usable.

---

### Part 3: The Master Prompt (Iteration 1)
*Copy and paste this into a high-level LLM (like GPT-4o or Claude 3.5 Sonnet) to start the "Product Requirements Document" (PRD) phase.*

> **Prompt:**
> "I want to develop a high-end AI application that transforms complex literary texts into personalized cinematic video series. 
>
> **The Core Workflow:** 
> 1. **Input:** The user provides a source text (e.g., 'Bhagavad Gita, Chapter 8').
> 2. **Casting:** The user uploads images of real people and assigns them roles (e.g., 'Person A is Krishna', 'Person B is Arjuna'). 
> 3. **Voice Synthesis:** The user uploads audio samples of these people to clone their voices.
> 4. **Production:** The app generates a script, generates AI video clips featuring the 'actors' with consistent faces, syncs the cloned voices to the lip-syncing, and compiles them into a video series.
>
> **Your Task:** 
> Act as a Product Manager and Senior Systems Architect. Based on this idea, please provide:
> 1. **User Journey Map:** From the first upload to the final video export.
> 2. **Technical Stack Recommendations:** What specific AI models (for Video, Lip-Sync, Voice Cloning, and LLM Scripting) should be integrated?
> 3. **Feature Roadmap:** Break this down into MVP (Minimum Viable Product), Version 2, and Version 3.
> 4. **Data Privacy & Ethics:** How should we handle the 'Face' and 'Voice' data to ensure it isn't used outside the app?
>
> **Note:** Do not write code yet. I want to refine the logic and the 'feel' of the app first."

---

### How to proceed from here:
1.  **Run the prompt above.**
2.  Look at the **Technical Stack** it gives you (it will likely mention things like *Stable Video Diffusion, ElevenLabs, or HeyGen API*).
3.  **Fine-tune:** Tell the AI, *"I want to make it easier for people who don't know how to write scripts—make the 'Storytelling' part more automated,"* or *"I want to make sure the 'Actors' look exactly like the photos—how do we solve for high-fidelity face swapping?"*
4.  **Repeat** until you have a blueprint you are happy with.
