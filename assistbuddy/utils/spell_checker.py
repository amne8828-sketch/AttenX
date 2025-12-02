from typing import Dict, List, Tuple
import re

try:
    from autocorrect import Speller
    AUTOCORRECT_AVAILABLE = True
except ImportError:
    AUTOCORRECT_AVAILABLE = False
    print("autocorrect not installed. Install with: pip install autocorrect")

try:
    import language_tool_python
    LANGUAGE_TOOL_AVAILABLE = True
except ImportError:
    LANGUAGE_TOOL_AVAILABLE = False
    print("language_tool_python not installed. Install with: pip install language-tool-python")


class SpellingCorrector:
    def __init__(self, language='en'):
        self.language = language
        self.custom_dictionary = set()
        
        # Initialize spellers
        if AUTOCORRECT_AVAILABLE:
            try:
                self.english_speller = Speller(lang='en')
            except:
                self.english_speller = None
                print("Failed to initialize English speller")
        else:
            self.english_speller = None
        
        # Initialize grammar tool
        if LANGUAGE_TOOL_AVAILABLE:
            try:
                self.grammar_tool = language_tool_python.LanguageTool(language)
            except:
                self.grammar_tool = None
                print(f"Failed to initialize grammar tool for {language}")
        else:
            self.grammar_tool = None

    def add_to_dictionary(self, words: List[str]):
        self.custom_dictionary.update(words)

    def correct_text(self, text: str, check_grammar: bool = True) -> Dict[str, any]:
        result = {
            'original': text,
            'corrected': text,
            'spelling_corrections': [],
            'grammar_corrections': [],
            'confidence': 1.0
        }
        
        # Step 1: Spell check with autocorrect
        if self.english_speller and AUTOCORRECT_AVAILABLE:
            try:
                corrected_text = self.english_speller(text)
                
                # Find differences
                original_words = text.split()
                corrected_words = corrected_text.split()
                
                for orig, corr in zip(original_words, corrected_words):
                    if orig != corr and orig.lower() not in self.custom_dictionary:
                        result['spelling_corrections'].append({
                            'original': orig,
                            'correction': corr,
                            'position': text.find(orig)
                        })
                
                result['corrected'] = corrected_text
            except Exception as e:
                print(f"Spell check error: {e}")
                result['corrected'] = text
        
        # Step 2: Grammar check
        if check_grammar and self.grammar_tool and LANGUAGE_TOOL_AVAILABLE:
            try:
                matches = self.grammar_tool.check(result['corrected'])
                
                if matches:
                    # Apply corrections
                    corrected_text = language_tool_python.utils.correct(
                        result['corrected'], 
                        matches
                    )
                    
                    # Store grammar corrections
                    for match in matches:
                        result['grammar_corrections'].append({
                            'message': match.message,
                            'suggestions': match.replacements[:3],  # Top 3 suggestions
                            'position': match.offset,
                            'length': match.errorLength,
                            'rule': match.ruleId
                        })
                    
                    result['corrected'] = corrected_text
                    result['confidence'] = max(0, 1.0 - (len(matches) * 0.05))  # Reduce confidence based on errors
            except Exception as e:
                print(f"Grammar check error: {e}")
        
        return result

    def correct_word(self, word: str) -> List[str]:
        if not self.english_speller:
            return [word]
        
        try:
            # Check if word is in custom dictionary
            if word.lower() in self.custom_dictionary:
                return [word]
            
            # Get correction
            corrected = self.english_speller(word)
            
            if corrected == word:
                return [word]
            
            return [corrected, word]  # Return both corrected and original
        except:
            return [word]

    def is_correctly_spelled(self, word: str) -> bool:
        if word.lower() in self.custom_dictionary:
            return True
        
        if not self.english_speller:
            return True  # Assume correct if speller not available
        
        try:
            corrected = self.english_speller(word)
            return corrected.lower() == word.lower()
        except:
            return True

    def highlight_errors(self, text: str) -> str:
        result = self.correct_text(text)
        highlighted = result['original']
        
        # Highlight spelling errors
        for error in result['spelling_corrections']:
            original = error['original']
            correction = error['correction']
            highlighted = highlighted.replace(
                original,
                f"[[{original}→{correction}]]"
            )
        
        return highlighted


# Global instance for easy import
_default_corrector = None


def get_corrector(language='en') -> SpellingCorrector:
    global _default_corrector
    if _default_corrector is None:
        _default_corrector = SpellingCorrector(language)
    return _default_corrector


def quick_correct(text: str) -> str:
    corrector = get_corrector()
    result = corrector.correct_text(text)
    return result['corrected']


# Example usage
if __name__ == "__main__":
    # Test the corrector
    corrector = SpellingCorrector()
    
    test_text = "Teh quik brown fox jumps ovr the lazy dog. Its a beautifull day!"
    
    result = corrector.correct_text(test_text)
    
    print("Original:", result['original'])
    print("Corrected:", result['corrected'])
    
    print("\nSpelling corrections:")
    for corr in result['spelling_corrections']:
        print(f"  {corr['original']} → {corr['correction']}")
    
    print("\nGrammar corrections:")
    for corr in result['grammar_corrections']:
        print(f"  {corr['message']}")
        print(f"  Suggestions: {', '.join(corr['suggestions'])}")