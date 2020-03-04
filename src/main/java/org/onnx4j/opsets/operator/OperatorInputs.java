/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.onnx4j.opsets.operator;

import java.util.LinkedList;
import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.opsets.operator.fields.InputField;

public abstract class OperatorInputs<T_TENSOR> {

	protected List<InputField<T_TENSOR>> inputFields = new LinkedList<InputField<T_TENSOR>>();
	protected Input[] inputArray;
	protected Attributes attrs;

	public OperatorInputs(Node node, Inputs inputs) {
		this.inputArray = inputs.get();
		this.attrs = node.getAttrs();
	}

	public <T extends OperatorInputs<T_TENSOR>> T cast(Class<T> clazz) {
	    return clazz.isInstance(this) ? clazz.cast(this) : null;
	}

	public void addInputField(InputField<T_TENSOR> field) {
		this.inputFields.add(field);
	}

	public Input[] getInputArray() {
		return inputArray;
	}

	public Attributes getAttrs() {
		return attrs;
	}
	
	public List<InputField<T_TENSOR>> getInputFields() {
		return this.inputFields;
	}

}