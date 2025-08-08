#!/usr/bin/env python3
"""
Example configuration for reference audio evaluation during adapter training.
This file shows how to set up evaluation references programmatically.
"""

# Example evaluation references configuration
EVAL_REFERENCES = [
    {
        'ref_audio': 'path/to/english_speaker.wav',
        'ref_text': 'This is a sample English reference text.',
        'gen_text': 'Hello, this is a test of the English adapter.',
        'language': 'en',
        'name': 'english_test'
    },
    {
        'ref_audio': 'path/to/romanian_speaker.wav',
        'ref_text': 'Aceasta este o referință românească.',
        'gen_text': 'Bună ziua, acesta este un test al adapterului român.',
        'language': 'ro',
        'name': 'romanian_test'
    },
    {
        'ref_audio': 'path/to/speaker_unknown_text.wav',
        'ref_text': '',  # Empty for automatic transcription
        'gen_text': 'This text will be generated using the voice from the reference.',
        'language': 'en',
        'name': 'auto_transcribed'
    }
]

# Command line equivalent:
COMMAND_LINE_ARGS = """
--eval_ref_audio path/to/english_speaker.wav path/to/romanian_speaker.wav path/to/speaker_unknown_text.wav
--eval_ref_text "This is a sample English reference text." "Aceasta este o referință românească." ""
--eval_gen_text "Hello, this is a test of the English adapter." "Bună ziua, acesta este un test al adapterului român." "This text will be generated using the voice from the reference."
--eval_languages en ro en
--eval_names english_test romanian_test auto_transcribed
--eval_steps 1000
"""

if __name__ == "__main__":
    print("Example evaluation references:")
    for i, ref in enumerate(EVAL_REFERENCES):
        print(f"\n{i+1}. {ref['name']} ({ref['language']})")
        print(f"   Audio: {ref['ref_audio']}")
        print(f"   Ref text: {ref['ref_text'] or '[AUTO-TRANSCRIBED]'}")
        print(f"   Gen text: {ref['gen_text']}")
    
    print(f"\nCommand line equivalent:")
    print(COMMAND_LINE_ARGS.strip()) 