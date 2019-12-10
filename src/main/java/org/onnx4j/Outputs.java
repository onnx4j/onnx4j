package org.onnx4j;

import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.onnx4j.Inputs.Input;
import org.onnx4j.model.graph.Node;
import org.onnx4j.utils.CastUtil;

public final class Outputs {

	public static class Output {

		private String name;
		private Object tensor;

		public static <T_TS> Output wrap(String name, T_TS tensor) {
			Output output = new Output();
			output.name = name;
			output.tensor = tensor;
			return output;
		}

		private Output() {
		}
		
		public String getName() {
			return this.name;
		}

		public <T_TS> T_TS getTensor(Class<T_TS> typeOfTensor) {
			return CastUtil.cast(this.tensor, typeOfTensor);
		}
		
		@SuppressWarnings("unchecked")
		public <T_TS> T_TS getTensor() {
			return (T_TS) this.tensor;
		}

	}

	private Node node;
	private Map<String, Output> outputs = new LinkedHashMap<String, Outputs.Output>();

	@SafeVarargs
	public static <T_TS> Outputs wrap(Node node, T_TS... tensors) {
		List<T_TS> tensorLst = new LinkedList<T_TS>();
		for (int n = 0; n < tensors.length; n++) {
			tensorLst.add(tensors[n]);
		}
		return Outputs.wrap(node, tensorLst);
	}
	public static <T_TS> Outputs wrap(Node node, List<T_TS> tensors) {
		Outputs outputList = new Outputs();
		outputList.node = node;
		for (int n = 0; n < tensors.size(); n++) {
			Output output = Output.wrap(node.getOutputNames()[n], tensors.get(n));
			outputList.outputs.put(node.getOutputNames()[n], output);
		}
		return outputList;
	}
	
	public void append(String name, Output output) {
		this.outputs.put(name, output);
	}
	
	public Inputs asInputs() {
		int convertIndex = 0;
		Input[] inputs = new Input[this.outputs.size()];
		for (Entry<String, Output> entry : this.outputs.entrySet()) {
			Input input = Input.wrap(entry.getKey(), this.node, entry.getValue());
			inputs[convertIndex++] = input;
		}
		return Inputs.wrap(inputs);
	}

	public Output get(String name) {
		return this.get(name);
	}

	public <T_TS> T_TS getTensor(String name) {
		Output output = this.outputs.get(name);
		if (output != null) {
			return output.getTensor();
		} else {
			return null;
		}
	}

	public Node getNode() {
		return this.node;
	}

	public Output[] get() {
		Output[] outputArray = new Output[this.outputs.size()];
		return this.outputs.values().toArray(outputArray);
	}
	
	public Map.Entry<String, Output> entrySet() {
		return new Map.Entry<String, Output>() {

			@Override
			public String getKey() {
				// TODO Auto-generated method stub
				return null;
			}

			@Override
			public Output getValue() {
				// TODO Auto-generated method stub
				return null;
			}

			@Override
			public Output setValue(Output value) {
				// TODO Auto-generated method stub
				return null;
			}
			
		};
	}

}
