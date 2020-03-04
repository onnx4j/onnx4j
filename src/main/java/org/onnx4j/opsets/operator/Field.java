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

import org.onnx4j.tensor.DataType;

public abstract class Field<T_TENSOR> {

	public static class TypeConstraint {

		private DataType[] dataTypes;

		public TypeConstraint(DataType... dataTypes) {
			this.dataTypes = dataTypes;
		}

		public DataType[] getDataTypes() {
			return dataTypes;
		}

	}

	protected T_TENSOR data;

	public Field(T_TENSOR data) {
		this.data = data;
	}

	public void setData(T_TENSOR data) {
		this.data = data;
	}

	public T_TENSOR getData() {
		return data;
	}

}
