package ml.learn.linear;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import ml.learn.object.Tag;

public class Template {
	public String template;
	public int[] relativePos;
	public int[] featureIdx;
	public  boolean isBigram;
	
	private String featureFormat;
	
	public Template(String template){
		this.template = template;
		compile();
	}
	
	private void compile(){
		Pattern macro = Pattern.compile("%x\\[(-?\\d+),(\\d+)\\]");
		Matcher matcher = macro.matcher(template.substring(1)); // Exclude the U or B
		int numMacros = 0;
		while(matcher.find()){
			numMacros += 1;
		}
		matcher = macro.matcher(template.substring(0));
		relativePos = new int[numMacros];
		featureIdx = new int[numMacros];
		int idx = 0;
		StringBuffer buf = new StringBuffer();
		isBigram = false;
		if(template.startsWith("B")){
			isBigram = true;
		}
		while(matcher.find()){
			relativePos[idx] = Integer.parseInt(matcher.group(1));
			featureIdx[idx] = Integer.parseInt(matcher.group(2));
			idx += 1;
			matcher.appendReplacement(buf, "%s");
		}
		matcher.appendTail(buf);
		featureFormat = buf.toString();
	}
	
	public static class TagIndex {
		public LinkedHashMap<Tag, Integer> unigramIndex;
		public LinkedHashMap<Tag, LinkedHashMap<Tag, Integer>> bigramIndex;
		
		public TagIndex(){
			unigramIndex = new LinkedHashMap<Tag, Integer>();
			bigramIndex = new LinkedHashMap<Tag, LinkedHashMap<Tag, Integer>>();
		}
		
		public void put(Tag prevTag, Tag curTag, int index, boolean isBigram){
			if(isBigram){
				if(!bigramIndex.containsKey(prevTag)){
					bigramIndex.put(prevTag, new LinkedHashMap<Tag, Integer>());
				}
				bigramIndex.get(prevTag).put(curTag, index);
			} else {
				unigramIndex.put(curTag, index);
			}
		}
		
		public int get(Tag prevTag, Tag curTag, boolean isBigram){
			if(isBigram){
				if(!bigramIndex.containsKey(prevTag)){
					return -1;
				}
				return bigramIndex.get(prevTag).getOrDefault(curTag, -1);
			} else {
				return unigramIndex.getOrDefault(curTag, -1);
			}
		}

		public String toString(){
			return "U"+unigramIndex.toString()+"B"+bigramIndex.toString();
		}
	}
	
	/**
	 * A class representing one class of feature.
	 * One class of feature consists of a set of possible tags (or tag sequences for bigram feature) that can 
	 * come with a specific feature (e.g., U00:Confidence, U12:NNP, B)
	 * Possible tags will be replacing the U (or B for bigram) with the tags' name to create a complete feature, 
	 * such as B-NP00:Confidence, B-NP:NNP, START|B-NP
	 * @author Aldrian Obaja <aldrianobaja.m@gmail.com>
	 *
	 */
	public static class Feature{
		/** The partially-filled feature name, except the Tag part (either U or B for unigram and bigram) */
		public String featureWithoutTag;
		/** The complete feature name this Feature object was created with */
		public String featureName;
		/** A mapping between full feature names of this Feature (represented by the tag names only) and the feature indices */
		public TagIndex tagIndex;
		/** Whether this feature is a bigram feature */
		public boolean isBigram;
		
		/**
		 * Create a new feature for the specified partial feature name with the specified tag sequence, 
		 * and whether this feature is a bigram feature
		 * @param feature
		 * @param prevTag
		 * @param curTag
		 * @param isBigram
		 */
		public Feature(String feature, Tag prevTag, Tag curTag, boolean isBigram){
			this.featureWithoutTag = feature;
			this.featureName = feature;
			this.isBigram = isBigram;
			if(isBigram){
				this.featureName = prevTag.text+"|"+curTag.text+this.featureName.substring(1);
			} else {
				this.featureName = curTag.text+this.featureName.substring(1);
			}
		}
		
		/**
		 * Whether the specified tag sequence is a possible sequence for current feature
		 * @param prevTag
		 * @param curTag
		 * @return
		 */
		public boolean present(Tag prevTag, Tag curTag){
			return tagIndex.get(prevTag, curTag, isBigram) != -1;
		}
		
		/**
		 * Adds the specified tag sequence as a possible sequence for current feature
		 * @param prevTag
		 * @param curTag
		 * @param index
		 */
		public void addTag(Tag prevTag, Tag curTag, int index){
			tagIndex.put(prevTag, curTag, index, isBigram);
		}
		
		/**
		 * Return the feature index of current feature with the specified tag sequence
		 * @param prevTag
		 * @param curTag
		 * @return
		 */
		public int getFeatureIndex(Tag prevTag, Tag curTag){
			return tagIndex.get(prevTag, curTag, isBigram);
		}
		
		public String toString(){
			StringBuilder builder = new StringBuilder();
			builder.append("["+featureWithoutTag+"]");
			builder.append("["+featureName+"]");
			builder.append(tagIndex);
			return builder.toString();
		}
	}
	
	/**
	 * Return the feature object at the specified position of the specified instance with the specified 
	 * tag sequence for current feature template
	 * @param instance
	 * @param position
	 * @param prevTag
	 * @param curTag
	 * @return
	 */
	public Feature getFeature(Instance instance, int position, Tag prevTag, Tag curTag){
		List<String> featureArguments = new ArrayList<String>();
		for(int i=0; i<relativePos.length; i++){
			featureArguments.add(instance.getFeatureAt(position+relativePos[i], featureIdx[i]));
		}
		return new Feature(String.format(featureFormat, featureArguments.toArray()), prevTag, curTag, isBigram);
	}
}
