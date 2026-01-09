#!/usr/bin/env python3
"""
English abbreviations example using pykokoro.

This example demonstrates how pykokoro handles common English abbreviations
including titles, time references, locations, measurements, and more.

Usage:
    python examples/abbreviations.py

Output:
    abbreviations_demo.wav - Generated speech with various abbreviations
"""

import soundfile as sf

import pykokoro

# Text with comprehensive abbreviations coverage
TEXT = """
Good morning! Let me introduce you to some people.

Meet Mr. Schmidt, Mrs. Johnson, Ms. Anderson, and Dr. Brown.
Prof. Williams and Rev. Martinez will join us at 3:00 p.m..
St. Patrick's Cathedral is located on 5th Ave. in New York, N.Y..

The meeting is scheduled for Mon., Jan. 15th at the company headquarters.
Please arrive by 9:30 a.m. and bring your I.D. card.

Our office is at 123 Main St., Apt. 4B, Washington, D.C., U.S.A..
For questions, contact us via email at info@example.com or call us ASAP.

The package weighs 5 lbs. and measures 10 ft. by 3 in..
The temperature reached 98°F, or approximately 37°C.

Lt. Commander Harris served in the U.S. Navy for 15 yrs..
He earned a Ph.D. in Computer Science from MIT in Sept. 2010.

The company, founded in 1995 A.D., operates in the U.K., Canada, etc..
Our CEO, Mr. Thompson Jr., will present the Q&A session.

Please R.S.V.P. by Fri., Dec. 1st.
P.S. Don't forget to bring your laptop!

Sincerely,
Dr. Emily Clarke, M.D.
Vice President, Research & Development
ABC Corp., Inc.
"""

VOICE = "af_bella"  # American Female voice
LANG = "en-us"  # American English


def main():
    """Generate English speech with abbreviations."""
    print("Initializing TTS engine...")
    kokoro = pykokoro.Kokoro()

    print("=== English Abbreviations Test ===")
    print(f"Voice: {VOICE}")
    print(f"Language: {LANG}\n")

    # Convert to phonemes to show how abbreviations are processed
    print("Converting text to phonemes...")
    phonemes = kokoro.phonemize(TEXT, lang=LANG)

    # Show a sample of the phonemes (first 500 chars)
    print("Sample phonemes (first 500 characters):")
    print(f"{phonemes[:500]}...\n")

    print(f"Total phoneme length: {len(phonemes)} characters\n")

    # List of abbreviations being tested
    abbreviations = [
        "Mr. (Mister)",
        "Mrs. (Missus)",
        "Ms. (Miss)",
        "Dr. (Doctor)",
        "Prof. (Professor)",
        "Rev. (Reverend)",
        "Lt. (Lieutenant)",
        "St. (Street/Saint)",
        "Ave. (Avenue)",
        "Apt. (Apartment)",
        "N.Y. (New York)",
        "D.C. (District of Columbia)",
        "U.S.A. (United States of America)",
        "U.K. (United Kingdom)",
        "Mon. (Monday)",
        "Jan. (January)",
        "Sept. (September)",
        "Dec. (December)",
        "Fri. (Friday)",
        "a.m. (ante meridiem)",
        "p.m. (post meridiem)",
        "I.D. (Identification)",
        "ASAP (As Soon As Possible)",
        "lbs. (pounds)",
        "ft. (feet)",
        "in. (inches)",
        "°F (degrees Fahrenheit)",
        "°C (degrees Celsius)",
        "yrs. (years)",
        "Ph.D. (Doctor of Philosophy)",
        "MIT (Massachusetts Institute of Technology)",
        "A.D. (Anno Domini)",
        "Jr. (Junior)",
        "CEO (Chief Executive Officer)",
        "Q&A (Questions and Answers)",
        "R.S.V.P. (Répondez s'il vous plaît)",
        "P.S. (Post Script)",
        "M.D. (Medical Doctor)",
        "Inc. (Incorporated)",
        "Corp. (Corporation)",
        "etc. (et cetera)",
    ]

    print("Abbreviations being tested:")
    for abbr in abbreviations:
        print(f"  - {abbr}")
    print()

    print("Generating audio with automatic splitting...")
    samples, sample_rate = kokoro.create(
        TEXT,
        voice=VOICE,
        speed=1.0,
        lang=LANG,
        trim_silence=True,
        split_mode="sentence",
    )

    output_file = "abbreviations_demo.wav"
    sf.write(output_file, samples, sample_rate)

    duration = len(samples) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")
    print("\nListen to verify that abbreviations are pronounced correctly!")

    kokoro.close()


if __name__ == "__main__":
    main()
