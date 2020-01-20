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
package org.onnx4j.model.graph;

import org.onnx4j.Model;
import org.onnx4j.NamedOnnxObject;
import org.onnx4j.Tensor;
import org.onnx4j.prototypes.OnnxProto3.TensorProto;
import org.onnx4j.tensor.TensorBuilder;

public class Constant extends NamedOnnxObject {

	private Tensor tensor;

	public Constant(Model model, TensorProto initializer) {
		super(initializer.getName(), initializer.getDocString());
		this.tensor = TensorBuilder
				.builder(initializer, model.getTensorOptions())
				.manager(model.getTensorManager())
				.build();
	}

	public Tensor getTensor() {
		return tensor;
	}

}