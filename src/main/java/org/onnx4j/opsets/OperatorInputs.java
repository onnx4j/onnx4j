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
package org.onnx4j.opsets;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.model.graph.Node;

public abstract class OperatorInputs {

	protected Input[] inputArray;

	public OperatorInputs(Node node, Inputs inputs) {
		this.inputArray = inputs.get();
	}
	
	public <T extends OperatorInputs> T cast(Class<T> clazz) {
	    return clazz.isInstance(this) ? clazz.cast(this) : null;
	}

	public Input[] getInputArray() {
		return inputArray;
	}

}