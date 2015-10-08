package ml.learn.object;

public class TaggedWord {
	private String word;
	private String[] features;
	private Tag tag;
	
	public TaggedWord(String word, Tag tag){
		this.setWord(word);
		this.setFeatures(new String[]{word});
		this.setTag(tag);
	}
	
	public TaggedWord(String[] features, Tag tag){
		this.setWord(features[0]);
		this.setFeatures(features);
		this.setTag(tag);
	}
	
	public String word() {
		return word;
	}

	public void setWord(String word) {
		this.word = word;
	}

	public Tag tag() {
		return tag;
	}

	public void setTag(Tag tag) {
		this.tag = tag;
	}

	public String[] features() {
		return features;
	}

	public void setFeatures(String[] features) {
		this.features = features;
	}
	
	public String toString(){
		return word()+"/"+tag();
	}

	public String conllString(){
		StringBuilder result = new StringBuilder();
		for(String feature: features()){
			result.append(feature+" ");
		}
		result.append(tag());
		return result.toString();
	}
}
