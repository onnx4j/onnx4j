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
package org.onnx4j.model.graph.node.attributes;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.onnx4j.Model;
import org.onnx4j.Tensor;
import org.onnx4j.model.graph.node.Attribute;
import org.onnx4j.prototypes.OnnxProto3.AttributeProto;
import org.onnx4j.prototypes.OnnxProto3.TensorProto;
import org.onnx4j.tensor.TensorBuilder;

public class TensorsAttribute extends Attribute<List<Tensor>> {

	public <T> TensorsAttribute(Model model, AttributeProto attrProto) {
		super(toTensors(model, attrProto), attrProto.getName(), attrProto.getDocString());
	}

	/**
	 * 由于是引用传递，这里返回一个不可修改的List对象，防止Operator在执行的过程中修改List对象的值。
	 */
	@Override
	public List<Tensor> getValue() {
		return Collections.unmodifiableList(super.getValue());
	}

	private static List<Tensor> toTensors(Model model, AttributeProto attrProto) {
		List<Tensor> tensors = new ArrayList<Tensor>();
		for (TensorProto tensorProto : attrProto.getTensorsList()) {
			Tensor tensor = TensorBuilder.builder(tensorProto, model.getTensorOptions())
					.manager(model.getTensorManager()).build();
			tensors.add(tensor);
		}
		return tensors;
	}

}