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
package org.onnx4j.opsets.operator.output;

import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.operator.Field.TypeConstraint;
import org.onnx4j.opsets.operator.OperatorOutputs;
import org.onnx4j.opsets.operator.fields.OutputField;

public abstract class SingleOperatorOutputs<T_TENSOR> extends OperatorOutputs<T_TENSOR> {
	
	public abstract TypeConstraint getTypeConstraint();

	public SingleOperatorOutputs(T_TENSOR output) {
		super.addOutputField(new OutputField<T_TENSOR>(this, getTypeConstraint(), output, false));
	}

	@Override
	public Outputs toOutputs(Node node) {
		return Outputs.wrap(node, this.outputFields.get(0).getData());
	}

}