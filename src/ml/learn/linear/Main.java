package ml.learn.linear;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;

import ml.learn.object.Tag;
import ml.learn.object.TaggedWord;

public class Main {
	public static void main(String[] args) throws Exception{
		boolean useCRF = true;
		boolean useCoNLLFormat = true;
//		boolean useCoNLLFormat = false;
		boolean printTestResult = false;
		boolean runInteractive = false;
		String trainingFile = "a2_data/sents.train";
		String testFile = "a2_data/sents.test";
//		String devFile = "a2_data/sents.devt";
		List<Instance> trainingData = readData(trainingFile, true);
		List<Instance> testData = readData(testFile, false);
//		List<Instance> devData = readData(devFile, true);
		
		String conllTrainingFile = "experiments/train.data";
		String conllTestFile = "experiments/test.data";
//		conllTrainingFile = "lihao data/training.pos";
//		conllTestFile = "lihao data/development.pos";
		if(useCoNLLFormat){
			trainingData = readCoNLLData(conllTrainingFile, true);
			testData = readCoNLLData(conllTestFile, true);
		}
		List<Instance> result;
//		List<Instance> reduced = new ArrayList<Instance>();
//		reduced.add(trainingData.get(1));
//		reduced.add(trainingData.get(3));
//		reduced = trainingData.subList(0, 40);
		StructuredClassifier classifier;
		if(useCRF){
			classifier = new CRF();
		} else {
			classifier = new HMM();
		}
//		testData = reduced;
		classifier.train(trainingData);
//		classifier.train(reduced);
		result = classifier.predict(testData);
//		result = classifier.predict(reduced);
		if(printTestResult){
			for(Instance instance: testData){
				System.out.println(instance);
			}
			for(Instance instance: result){
				System.out.println(instance);
			}
		}
		printScore(result, testData);
		if(useCoNLLFormat){
			FileWriter wr = new FileWriter("experiments/myresult");
			Iterator<Instance> predIter = result.iterator();
			Iterator<Instance> actuIter = testData.iterator();
			while(predIter.hasNext()){
				Instance predInst = predIter.next();
				Instance actuInst = actuIter.next();
				Iterator<TaggedWord> predWordIter = predInst.words.iterator();
				Iterator<TaggedWord> actuWordIter = actuInst.words.iterator();
				while(predWordIter.hasNext()){
					TaggedWord predWord = predWordIter.next();
					TaggedWord actuWord = actuWordIter.next();
					wr.write(actuWord.conllString()+" "+predWord.tag()+"\n");
				}
				wr.write("\n");
			}
			wr.close();
		}
		if(runInteractive){
			Scanner sc = new Scanner(System.in);
			while(true){
				System.out.print(">>> ");
				String line = sc.nextLine();
				if(line.length() == 0) break;
				String[] words = line.split(" ");
				String[] tags = new String[words.length];
				result = classifier.predict(Arrays.asList(new Instance[]{new Instance(words, tags)}));
				for(Instance instance: result){
					System.out.println(instance);
				}
			}
			sc.close();
		}
	}
	
	private static void printScore(List<Instance> predicted, List<Instance> actual){
		int total = 0;
		int correct = 0;
		Iterator<Instance> predIter = predicted.iterator();
		Iterator<Instance> actuIter = actual.iterator();
		while(predIter.hasNext()){
			Instance predInstance = predIter.next();
			Instance actuInstance = actuIter.next();
			Iterator<TaggedWord> predInstIter = predInstance.words.iterator();
			Iterator<TaggedWord> actuInstIter = actuInstance.words.iterator();
			while(predInstIter.hasNext()){
				total++;
				TaggedWord predWord = predInstIter.next();
				TaggedWord actuWord = actuInstIter.next();
				if(predWord.tag() == actuWord.tag()){
					correct++;
				}
			}
		}
		System.out.println(String.format("Accuracy = %d/%d = %.2f%%", correct, total, 100.0*correct/total));
	}
	
	private static List<Instance> readData(String fileName, boolean withTags) throws IOException{
		InputStreamReader isr = new InputStreamReader(new FileInputStream(fileName), "UTF-8");
		BufferedReader br = new BufferedReader(isr);
		List<Instance> result = new ArrayList<Instance>();
		while(br.ready()){
			String line = br.readLine();
			String[] wordTags = line.split(" ");
			String[] words = new String[wordTags.length];
			String[] tags = new String[wordTags.length];
			for(int i=0; i<wordTags.length; i++){
				if(withTags){
					int lastSlash = wordTags[i].lastIndexOf('/');
					words[i] = wordTags[i].substring(0,  lastSlash);
					tags[i] = wordTags[i].substring(lastSlash+1);
//					switch(tags[i].charAt(0)){
//					case 'N':
//					case 'P':
//						tags[i] = "N"; break;
//					case 'V':
//						tags[i] = "V"; break;
//					case 'A':
//					case 'R':
//						tags[i] = "A"; break;
//					case 'I':
//					case 'T':
//						tags[i] = "T"; break;
//					default: tags[i] = "O";
//					}
				} else {
					words[i] = wordTags[i];
					tags[i] = "";
				}
			}
			result.add(new Instance(words, tags));
		}
		br.close();
		return result;
	}
	
	private static List<Instance> readCoNLLData(String fileName, boolean withTags) throws IOException{
		InputStreamReader isr = new InputStreamReader(new FileInputStream(fileName), "UTF-8");
		BufferedReader br = new BufferedReader(isr);
		List<Instance> result = new ArrayList<Instance>();
		List<TaggedWord> words = null;
		while(br.ready()){
			if(words == null){
				words = new ArrayList<TaggedWord>();
			}
			String line = br.readLine().trim();
			if(line.length() == 0){
				result.add(new Instance(words));
				words = null;
			} else {
				int lastSpace = line.lastIndexOf(" ");
				String[] features = line.substring(0, lastSpace).split(" ");
				Tag tag = Tag.get(line.substring(lastSpace+1));
				words.add(new TaggedWord(features, tag));
			}
		}
		br.close();
		return result;
	}
}
