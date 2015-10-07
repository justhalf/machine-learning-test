package ml.learn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A class representing a single data point (a sequence of words and tags)
 * @author Aldrian Obaja <aldrianobaja.m@gmail.com>
 *
 */
public class Instance {
	/** List of words with their respective tags */
	public List<TaggedWord> words;
	private TaggedWord START = null;
	private TaggedWord END = null;
	
	public Instance(List<TaggedWord> words){
		this.words = new ArrayList<TaggedWord>(words.size());
		for(TaggedWord word: words){
			this.words.add(word);
		}
		initialize();
	}
	
	public Instance(String[] words, String[] tags){
		this.words = new ArrayList<TaggedWord>();
		for(int i=0; i<words.length; i++){
			this.words.add(new TaggedWord(words[i], Tag.get(tags[i])));
		}
		initialize();
	}
	
	private void initialize(){
		if(words.size() > 0){
			TaggedWord word = words.get(0);
			String[] startFeatures = new String[word.features().length];
			Arrays.fill(startFeatures, Tag.START.text);
			START = new TaggedWord(startFeatures, Tag.START);
			String[] endFeatures = new String[word.features().length];
			Arrays.fill(endFeatures, Tag.END.text);
			END = new TaggedWord(endFeatures, Tag.END);
		} else {
			START = new TaggedWord(Tag.START.text, Tag.START);
			END = new TaggedWord(Tag.END.text, Tag.END);
		}
	}
	
	public String getFeatureAt(int position, int featureIdx){
		TaggedWord taggedWord = getWordAt(position);
		return taggedWord.features()[featureIdx];
	}
	
	public TaggedWord getWordAt(int position){
		if(position < 0){
			return START;
		}
		if(position >= words.size()){
			return END;
		}
		return words.get(position);
	}
	
	public Tag getTagAt(int position){
		if(position < 0){
			return Tag.START;
		}
		if(position >= words.size()){
			return Tag.END;
		}
		return words.get(position).tag();
	}
	
	public String toString(){
		StringBuilder result = new StringBuilder();
		for(TaggedWord word: words){
			if(result.length() > 0) result.append(" ");
			result.append(word.toString());
		}
		return result.toString();
	}
}
