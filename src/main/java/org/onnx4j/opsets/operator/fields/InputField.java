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
package org.onnx4j.opsets.operator.fields;

import org.onnx4j.Inputs.Input;
import org.onnx4j.opsets.operator.Field;
import org.onnx4j.opsets.operator.OperatorInputs;

public class InputField<T_TENSOR> extends Field<T_TENSOR> {

	private TypeConstraint constraint;

	public InputField(OperatorInputs<T_TENSOR> operatorInputs, TypeConstraint constraint, Input input) {
		super(input.getTensor());
		this.constraint = constraint;
		operatorInputs.addInputField(this);
	}

	public TypeConstraint getConstraints() {
		return constraint;
	}

}
