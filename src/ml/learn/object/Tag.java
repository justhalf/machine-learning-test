package ml.learn.object;

import java.util.LinkedHashMap;

public class Tag {
	
	public static final LinkedHashMap<String, Tag> TAG_CACHE = new LinkedHashMap<String, Tag>();
	
	/** Indicating start of the sequence */
	public static final Tag START = Tag.create("START", true, false, false);
	/** Indicating end of the sequence */
	public static final Tag END = Tag.create("END", false, true, false);
	
	public String text;
	public boolean isStartTag;
	public boolean isEndTag;
	public boolean isOpenClass;
	
	public static Tag get(String tagName){
		if(!TAG_CACHE.containsKey(tagName)){
			TAG_CACHE.put(tagName, new Tag(tagName));
		}
		return TAG_CACHE.get(tagName);
	}
	
	public static Tag create(String tagName, boolean isStartTag, boolean isEndTag, boolean isOpenClass){
		if(TAG_CACHE.containsKey(tagName)){
			throw new IllegalArgumentException(String.format("The tag %s already exists!", tagName));
		}
		TAG_CACHE.put(tagName, new Tag(tagName, isStartTag, isEndTag, isOpenClass));
		return TAG_CACHE.get(tagName);
	}
	
	private Tag(String text, boolean isStartTag, boolean isEndTag, boolean isOpenClass){
		this.text = text;
		this.isStartTag = isStartTag;
		this.isEndTag = isEndTag;
		this.isOpenClass = isOpenClass;
	}
	
	public Tag(String text){
		this(text, false, false, true);
	}
	
	public String toString(){
		return text;
	}
	
	public boolean equals(Object o){
		if(o instanceof Tag){
			Tag t = (Tag)o;
			return t.text.equals(this.text);
		}
		return false;
	}
	
	public int hashCode(){
		return text.hashCode();
	}
}
