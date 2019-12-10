package org.onnx4j;

import java.util.LinkedHashMap;
import java.util.Map;

import org.onnx4j.model.graph.Node;
import org.onnx4j.utils.CastUtil;

public final class Inputs {

	public static class Input {

		private String name;
		private Node node;
		private Object tensor;

		public static <T_TS> Input wrap(String name, Node node, T_TS tensor) {
			Input input = new Input();
			input.node = node;
			input.name = name;
			input.tensor = tensor;
			return input;
		}

		private Input() {
		}

		public Node getNode() {
			return this.node;
		}

		public <T_TS> T_TS getTensor(Class<T_TS> typeOfTensor) {
			return CastUtil.cast(this.tensor, typeOfTensor);
		}

		@SuppressWarnings("unchecked")
		public <T_TS> T_TS getTensor() {
			return (T_TS) this.tensor;
		}

	}

	private Map<String, Input> inputs = new LinkedHashMap<String, Inputs.Input>();

	public static Inputs wrap(Input... inputs) {
		Inputs inputList = new Inputs();
		for (Input input : inputs) {
			inputList.inputs.put(input.name, input);
		}
		return inputList;
	}
	
	public void append(Input input) {
		this.inputs.put(input.name, input);
	}

	public Input get(String name) {
		return this.get(name);
	}

	public <T_TS> T_TS getTensor(String name, Class<T_TS> typeOfTensor) {
		Input input = this.inputs.get(name);
		if (input != null) {
			return input.getTensor(typeOfTensor);
		} else {
			return null;
		}
	}

	public Node getNode(String name) {
		Input input = this.inputs.get(name);
		if (input != null) {
			return input.getNode();
		} else {
			return null;
		}
	}

	public Input[] get() {
		Input[] inputArray = new Input[this.inputs.size()];
		return this.inputs.values().toArray(inputArray);
	}

}
