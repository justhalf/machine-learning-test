package ml.learn;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class Main {
	public static void main(String[] args) throws Exception{
		String trainingFile = "a2_data/sents.train";
		String testFile = "a2_data/sents.test";
		String devFile = "a2_data/sents.devt";
		List<Instance> trainingData = readData(trainingFile, true);
		List<Instance> testData = readData(testFile, false);
		List<Instance> devData = readData(devFile, true);
		List<Instance> result;
		List<Instance> reduced = new ArrayList<Instance>();
		reduced.add(trainingData.get(1));
		reduced.add(trainingData.get(3));
		List<Instance> reduced400 = trainingData.subList(0, 400);
		if(true){
			CRF crf = new CRF();
//			crf.train(trainingData);
			crf.train(reduced400);
			result = crf.predict(devData);
//			result = crf.predict(reduced);
		} else {
			HMM hmm = new HMM();
			hmm.train(trainingData);
			result = hmm.predict(testData);
		}
//		for(Instance instance: reduced){
//			System.out.println(instance);
//		}
//		for(Instance instance: result){
//			System.out.println(instance);
//		}
//		result = hmm.predict(devData);
		printScore(result, devData);
//		Scanner sc = new Scanner(System.in);
//		while(true){
//			System.out.print(">>> ");
//			String line = sc.nextLine();
//			if(line.length() == 0) break;
//			String[] words = line.split(" ");
//			String[] tags = new String[words.length];
//			result = hmm.predict(Arrays.asList(new Instance[]{new Instance(words, tags)}));
//			for(Instance instance: result){
//				System.out.println(instance);
//			}
//		}
//		sc.close();
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
					switch(tags[i].charAt(0)){
					case 'N':
					case 'P':
						tags[i] = "N"; break;
					case 'V':
						tags[i] = "V"; break;
					case 'A':
					case 'R':
						tags[i] = "A"; break;
					case 'I':
					case 'T':
						tags[i] = "T"; break;
					default: tags[i] = "O";
					}
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
}
